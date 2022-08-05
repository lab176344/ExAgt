import logging
from os import device_encoding

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.train.train import Training
from src.utils.average_meter import AverageMeter
from tqdm import tqdm
from torch.autograd import Variable

def prep_data(input_data, cuda):
    """
    Takes a batch of tuplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    input_data_out = dict((k, Variable(v)) for k,v in input_data.items())
    input_data = input_data_out
    
    if cuda:
        input_data_out = dict((k, v.cuda()) for k,v in input_data.items())
    return input_data_out

class train_100(Training):
    def __init__(self,
                 idx=2,
                 crops_for_assignment=None,
                 nmb_crops=None,
                 temperature=0.1,
                 freeze_prototypes_niters=313,
                 epsilon=0.05,
                 queue=None,
                 sinkhorn_iterations=3,
                 **kwargs) -> None:
        super().__init__()
        if nmb_crops is None:
            nmb_crops = [2]
        if crops_for_assignment is None:
            crops_for_assignment = [0, 1]
        self.description = "Double SWAV + RECON"
        self.temperature = temperature
        self.freeze_prototypes_niters = freeze_prototypes_niters
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations #--> loss def?
        self.queue = queue

    def run_training(self, model, dataloader_train, loss_fc, optimizer, _):
        self._train(model, dataloader_train, loss_fc, optimizer)

    def _train(self, model, dataloader_train, loss_fc, optimizer):
        epochs = dataloader_train.epochs
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        for epoch in range(epochs):
            loss_record = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch, rank=0)
            for batch_idx, input_data in enumerate(dataloader,start=epoch * len(dataloader)):
                
                input_data = prep_data(input_data, device)

                # normalize the prototypes
                with torch.no_grad():
                    model.normalize_prototypes()

                #TODO: image_in = torch.cat([x1, x2])??
                output_a  = model(input_data['anchor_image'],input_data['anchor_trajectory'])
                output_pp = model(input_data['pp_image'],input_data['pp_trajectory'])
                output_pn = model(input_data['pn_image'],input_data['pn_trajectory'])
                output_all = {'z_merge_a':output_a['z'],
                            'z_merge_pp' :output_pp['z'],
                            'z_infra_a'  :output_a['z_infra'],
                            'z_infra_pp' :output_pp['z_infra'],
                            'z_infra_pn' :output_pn['z_infra'],
                            'image_target_a':input_data['anchor_target'],
                            'image_out_a'   :output_a['x'],
                            'image_target_pp':input_data['pp_target'],
                            'image_out_pp'   :output_pp['x'],
                            'image_target_pn':input_data['pn_target'],
                            'image_out_pn'   :output_pn['x']}
                batch_pass += 1

                loss = loss_fc(output_all)

                # ============ backward and optim step ... ============
                loss.backward()
                # cancel gradients for the prototypes
                if batch_pass < self.freeze_prototypes_niters:
                    for name, p in model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None
                optimizer.step()

                pbar.update(1)
                loss_record.update(loss.item(), output_a['z'].size(0))
                log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f}".format(
                    epoch + 1, epochs, batch_pass, len(dataloader),
                    round(loss_record.avg, 3)).center(50)

                pbar.set_description(log_msg)
                logging.info(log_msg)


