import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from src.train.train import Training
from src.utils.average_meter import AverageMeter
from tqdm import tqdm
import os

class train_1(Training):
    def __init__(self,
                 idx=2,
                 crops_for_assignment=None,
                 nmb_crops=None,
                 temperature=0.1,
                 freeze_prototypes_niters=313,
                 epsilon=0.05,
                 sinkhorn_iterations=3,
                 **kwargs) -> None:
        super().__init__()
        if nmb_crops is None:
            nmb_crops = [2]
        if crops_for_assignment is None:
            crops_for_assignment = [0, 1]
        self.nmb_crops = nmb_crops
        self.description = "Swav training logic"
        self.crops_for_assignment = crops_for_assignment
        self.temperature = temperature
        self.freeze_prototypes_niters = freeze_prototypes_niters
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.queue = None


        self.queue_length = 100
        self.epoch_queue_starts = 0
        self.use_the_queue = True


    def run_training(self, model, dataloader_train, loss_fc, optimizer, scheduler):
        mp.spawn(self._train,
                 nprocs=dataloader_train.num_gpus,
                 args=(model, dataloader_train, loss_fc, optimizer, scheduler))


    def _train(self, rank, model, dataloader_train, loss_fc, optimizer, scheduler):
        assert dist.is_nccl_available()

        epochs = dataloader_train.epochs
        world_size = dataloader_train.num_gpus
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset[0]) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))

        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=world_size,
                                rank=rank)
        device = torch.device(rank)
        torch.cuda.set_device(device)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank])

        model.train()

        # build the queue
        self.queue = None
        queue_path = os.path.join("./temp/",
                                  "queue" + str(rank) + ".pth")
        if os.path.isfile(queue_path):
            self.queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        self.queue_length -= self.queue_length % (
                    dataloader_train.batch_size * world_size)

        for epoch in range(epochs):
            loss_record = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch, rank=rank)

            # optionally starts a queue
            if self.queue_length > 0 and epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assignment),
                    self.queue_length // world_size,
                    model.module.feature_dim,
                ).cuda()

            for batch_idx, (x1, x2, y,
                            idx) in enumerate(dataloader,
                                              start=epoch * len(dataloader)):
                # update learning rate
                if scheduler is not None:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = scheduler.get_lr_for_iter(batch_idx)

                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                # normalize the prototypes
                with torch.no_grad():
                    w = model.module.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    model.module.prototypes.weight.copy_(w)

                # ============ multi-res forward passes ... ============
                inputs = torch.cat([x1, x2])
                embedding, output = model(inputs)
                embedding = embedding.detach()
                bs = inputs[0].size(0)
                batch_pass += 1

                # ============ swav loss ... ============
                loss = 0
                for i, crop_id in enumerate(self.crops_for_assignment):
                    with torch.no_grad():
                        out = output[bs * crop_id:bs * (crop_id + 1)].detach()

                        # time to use the queue
                        if self.queue is not None:
                            if self.use_the_queue or not torch.all(
                                    self.queue[i, -1, :] == 0):
                                self.use_the_queue = True
                                out = torch.cat((torch.mm(
                                    self.queue[i],
                                    model.module.prototypes.weight.t()), out))
                            # fill the queue
                            self.queue[i, bs:] = self.queue[i, :-bs].clone()
                            self.queue[i, :bs] = embedding[crop_id *
                                                           bs:(crop_id + 1) *
                                                           bs]

                        # get assignments
                        q = self.distributed_sinkhorn(
                            out, dataloader_train.num_gpus)[-bs:]

                    # cluster assignment prediction
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(self.nmb_crops)),
                                       crop_id):
                        x = output[bs * v:bs * (v + 1)] / self.temperature
                        subloss -= torch.mean(
                            torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(self.nmb_crops) - 1)
                loss /= len(self.crops_for_assignment)

                # ============ backward and optim step ... ============
                loss.backward()
                # cancel gradients for the prototypes
                if batch_idx < self.freeze_prototypes_niters:
                    for name, p in model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None
                optimizer.step()
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                if rank == 0:
                    pbar.update(world_size)
                    loss_record.update(loss.item(), x1.size(0))
                    log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f} Lr: {:.5f}".format(
                        epoch + 1, epochs, batch_pass, len(dataloader),
                        round(loss_record.avg, 3),
                        optimizer.param_groups[0]["lr"]).center(50)

                    pbar.set_description(log_msg)
                    logging.info(log_msg)

    @torch.no_grad()
    def distributed_sinkhorn(self, out, world_size):
        Q = torch.exp(out / self.epsilon).t(
        )  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
