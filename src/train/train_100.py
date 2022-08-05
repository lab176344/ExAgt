import logging
from os import device_encoding
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.train.train import Training
from src.utils.average_meter import AverageMeter
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

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

    def run_training(self, model, dataloader_train, loss_fc, optimizer, _,dataloader_test):
        self._train(model, dataloader_train, loss_fc, optimizer,dataloader_test)

    def _train(self, model, dataloader_train, loss_fc, optimizer,dataloader_test):
        epochs = dataloader_train.epochs
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        writer = SummaryWriter()
        dataloader_test = dataloader_test()
        z_test = torch.zeros((len(dataloader_test.dataset),model.z_dim_m))
        labels_test = []*len(dataloader_test.dataset)
        for epoch in range(epochs):
            loss_record = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch, rank=0)
            for batch_idx, input_data in enumerate(dataloader,start=epoch * len(dataloader)):
                # Save the projection
                if batch_idx % 6000 == 0:
                    for batch_idx_test,proj_data in enumerate(dataloader_test):
                        image_in = proj_data[0].cuda()
                        traj_in = (proj_data[1].cuda(),proj_data[2])
                        output_test  = model(image_in,traj_in,return_infra_cXz=True,return_infra_z=True,return_infra_q=True,return_merge_cXz=True,return_merge_q=True)
                        z_test[(batch_idx_test)*dataloader_test.batch_size:(batch_idx_test+1)*dataloader_test.batch_size,:] = output_test['z'].detach().cpu()
                        labels_test[(batch_idx_test)*dataloader_test.batch_size:(batch_idx_test+1)*dataloader_test.batch_size] = [int(val) for val in proj_data[4]]
                    writer.add_embedding(z_test, metadata=labels_test,global_step=batch_idx)
                
                
                
                #input_data = prep_data(input_data, device)

                # normalize the prototypes
                with torch.no_grad():
                    model.normalize_prototypes()

                output_a  = model(input_data[0][0].cuda(),(input_data[0][1].cuda(),input_data[0][2]),return_infra_cXz=True,return_infra_z=True,return_infra_q=True,return_merge_cXz=True,return_merge_q=True)
                output_pp = model(input_data[1][0].cuda(),(input_data[1][1].cuda(),input_data[1][2]),no_recon=True,return_infra_cXz=True,return_infra_z=True,return_infra_q=True,return_merge_cXz=True,return_merge_q=True)
                output_pn = model(input_data[2][0].cuda(),(input_data[2][1].cuda(),input_data[2][2]),no_recon=True,return_infra_cXz=True,return_infra_z=True,return_infra_q=True,return_merge_cXz=True,return_merge_q=True)
                output_all = {'cXz_infra_a'  :output_a['cXz_infra'],
                            'cXz_infra_pp' :output_pp['cXz_infra'],
                            'cXz_infra_pn' :output_pn['cXz_infra'],
                            'cXz_merge_a'  :output_a['cXz_merge'],
                            'cXz_merge_pp' :output_pp['cXz_merge'],
                            'cXz_merge_pn' :output_pn['cXz_merge'],
                            'image_target_a':input_data[0][3].cuda(),
                            'image_out_a'   :output_a['x']}
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
                writer.add_scalar("Train Loss", loss, batch_idx)
                writer.add_scalar("Train Loss - AVG", loss_record.avg, batch_idx)
                logging.info(log_msg)


