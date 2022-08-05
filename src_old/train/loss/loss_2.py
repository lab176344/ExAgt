import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from src.train.loss.loss import loss


class loss_2(loss):
    def __init__(self, idx=2, name="swav", description="todo", input_="todo", output="todo",
                 crops_for_assignment=[0, 1], nmb_crops=[2], temperature=0.1,
                 freeze_prototypes_niters=313, epsilon=0.05,
                 sinkhorn_iterations=3, world_size=1):

        super().__init__(idx, name, description, input_, output)
        raise NotImplementedError("unused")
        self.crops_for_assigment = crops_for_assignment
        self.nmb_crops = nmb_crops
        self.temperature = temperature
        self.freeze_prototypes_niters = freeze_prototypes_niters
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.world_size = world_size

    def forward(self,embedding, output, model,queue=None, freeze_prototypes=False):
        use_the_queue = False
        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = model(inputs)
        embedding = embedding.detach()
        bs = embedding[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assigment):
            with torch.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i,
                                                            -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat(
                            (torch.mm(queue[i],
                                      model.module.prototypes.weight.t()),
                             out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) *
                                              bs]

                # get assignments
                q = self.distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v:bs * (v + 1)] / self.temperature
                subloss -= torch.mean(
                    torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assigment)

        # ============ backward and optim step ... ============
        # cancel gradients for the prototypes
        if freeze_prototypes:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

        # ============ misc ... ============
        # if args.rank == 0 and it % 50 == 0:
        #     logger.info(
        #         "Epoch: [{0}][{1}]\t"
        #         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        #         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
        #         "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
        #         "Lr: {lr:.4f}".format(
        #             epoch,
        #             it,
        #             batch_time=batch_time,
        #             data_time=data_time,
        #             loss=losses,
        #             lr=optimizer.optim.param_groups[0]["lr"],
        #         )
        #     )
        return loss

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t(
        )  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.world_size  # number of samples to assign
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
