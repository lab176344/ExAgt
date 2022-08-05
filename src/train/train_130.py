import logging
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.train.train import Training
from src.utils.average_meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.autograd import Variable
from einops import rearrange, reduce, repeat

def prep_data(input_data,cuda):
    """
    Takes a batch of tuplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    input_data_out = dict((k, Variable(v)) for k,v in input_data.items())
    input_data = input_data_out
    
    if cuda:
        input_data_out = dict((k, v.cuda()) for k,v in input_data.items())
    return input_data_out

class train_130(Training):
    def __init__(self, idx=130, **kwargs) -> None:
        super().__init__()
        self.description = "TODO"

    def run_training(self, model, dataloader_train, loss_fc, optimizer,
                     scheduler):
        if dataloader_train.num_gpus > 1:
            mp.spawn(self._train_dist,
                     nprocs=dataloader_train.num_gpus,
                     args=(model, dataloader_train, loss_fc, optimizer))
        else:
            self._train(model, dataloader_train, loss_fc, optimizer)

    @staticmethod
    def _train(model, dataloader_train, loss_fc, optimizer) -> None:
        """Single gpu training
        """
        epochs = dataloader_train.epochs
        writer = SummaryWriter()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        time_start = datetime.now()
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        model.train()
        model.to(device)
        for epoch in range(epochs):
            loss_record = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch)
            for batch_idx, sample in enumerate(dataloader,
                                              start=epoch * len(dataloader)):
                batch_pass = batch_pass + 1
               
                sample = prep_data(sample, cuda=True)
                sample_hist = sample['hist_ego'].float()
                #sample = rearrange(sample, 'b t f -> b f t')
                gTruth = sample['pred_ego'].float()

                pred = model(sample_hist,gTruth)
               
                #gTruth = rearrange(gTruth, 'b t f -> b f t')
                              
                loss = loss_fc(pred,gTruth)

                loss_record.update(loss.item(), sample_hist.size(0))
                # compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f}".format(
                    epoch + 1, epochs, batch_pass, len(dataloader),
                    round(loss_record.avg, 3)).center(50)
                pbar.set_description(log_msg)
                writer.add_scalar("Train Loss", loss, batch_idx)
                logging.info(log_msg)

        pbar.set_description("Training finished {} (Total time: {})".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            datetime.now() - time_start).center(50))

        pbar.close()
        logging.info(
            "End of Training -- total time: {}".format(datetime.now() -
                                                       time_start))
    #TODO
    @staticmethod
    def _train_dist(rank, model, dataloader_train, loss_fc, optimizer) -> None:
        assert dist.is_nccl_available()
        #TODO merge with _train
        pass
        # writer = SummaryWriter()
        # world_size = dataloader_train.num_gpus
        # epochs = dataloader_train.epochs
        # pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                            #   (dataloader_train.batch_size)),
                    # desc="init training...".center(50))

        # dist.init_process_group(backend='nccl',
                                # init_method='env://',
                                # world_size=world_size,
                                # rank=rank)

        # device = torch.device(rank)
        # model.to(device)
        # if dataloader_train.batch_size < 4:
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # model = torch.nn.parallel.DistributedDataParallel(
            # model, broadcast_buffers=False, device_ids=[rank])
        # model.train()
        # for epoch in range(epochs):
            # loss_record = AverageMeter()
            # batch_pass = 0
            # dataloader = dataloader_train(epoch, rank)
            # for batch_idx, (x1, x2, y,
                            # idx) in enumerate(dataloader,
                                            #   start=epoch * len(dataloader)):
                # batch_pass = batch_pass + 1
                # x1, x2 = x1.to(device,
                            #    non_blocking=True), x2.to(device,
                                                        #  non_blocking=True)
                # output1, feature1 = model(x1)
                # output2, feature2 = model(x2)
                # loss = loss_fc(output1, output2, device=device)

                #compute gradient and do optimizer step
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # dist.all_reduce(loss, dist.ReduceOp.SUM)
                # loss_record.update(loss.item() / world_size,
                                #    x1.size(0) * world_size)
                # if rank == 0:
                    # pbar.update(world_size)
                    # log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f}".format(
                        # epoch + 1, epochs, batch_pass * world_size,
                        # len(dataloader) * world_size,
                        # round(loss_record.avg, 3)).center(50)
                    # pbar.set_description(log_msg)
                    # writer.add_scalar("Train Loss", loss / world_size,
                                    #   batch_idx)
                    # logging.info(log_msg)
