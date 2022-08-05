import logging
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.utils.average_meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class train_0():
    def __init__(self) -> None:
        pass

    def run_training(self, model, dataloader_train, loss_fc, optimizer):
        self._train(model, dataloader_train, loss_fc, optimizer)

    @staticmethod
    def _train(model, dataloader_train, loss_fc, optimizer) -> None:
        epochs = dataloader_train.epochs
        writer = SummaryWriter()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        time_start = datetime.now()
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset[0]) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        model.train()
        model.to(device)
        for epoch in range(epochs):
            loss_record = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch)
            for batch_idx, (x1, x2, y,
                            idx) in enumerate(dataloader,
                                              start=epoch * len(dataloader)):
                batch_pass = batch_pass + 1
                x1, x2 = x1.to(device, non_blocking=True).float(), x2.to(
                    device, non_blocking=True).float()
                embedding1, _ = model(x1)
                embedding2, _ = model(x2)
                check = int((embedding1 != embedding1).sum())
                if (check > 0):
                    print("your data contains Nan")
                loss = loss_fc(embedding1,
                               embedding2,
                               device=device,
                               world_size=dataloader_train.num_gpus)

                loss_record.update(loss.item(), x1.size(0))
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
                return

        pbar.set_description("Training finished {} (Total time: {})".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            datetime.now() - time_start).center(50))

        pbar.close()
        logging.info(
            "End of Training -- total time: {}".format(datetime.now() -
                                                       time_start))