import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.train.train import Training
from tqdm import tqdm


class train_2(Training):
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
        self.nmb_crops = nmb_crops
        self.description = "Swav training logic"
        self.crops_for_assigment = crops_for_assignment
        self.temperature = temperature
        self.freeze_prototypes_niters = freeze_prototypes_niters
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.queue = queue

    def run_training(self, model, dataloader_train, loss_fc, optimizer):
        self._train(model, dataloader_train, loss_fc, optimizer)

    def _train(self, model, dataloader_train, loss_fc, optimizer):
        epochs = dataloader_train.epochs
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        for epoch in range(epochs):
            dataloader = dataloader_train(epoch, rank=0)
            for batch_idx, (F_x, F_y, F_vx, F_vy,
                            xy_pred) in enumerate(dataloader,
                                              start=epoch * len(dataloader)):
                v0 = F_vx[0,-1]
                y_hat, z, F_re = model(F_x.transpose(0,1).float().to(device), F_y.transpose(0,1).float().to(device), 
                                       F_vx.transpose(0,1).float().to(device), F_vy.transpose(0,1).float().to(device), v0.float().to(device))
                loss = loss_fc(torch.cat((xy_pred[0, 0, :], xy_pred[0, 1, :])).to(device).float(), y_hat[0].to(device))
                # ========================================================= backward
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Getting gradients w.r.t. parameters
                loss.backward()      
                print(loss.item())
                # Updating parameters
                optimizer.step()                
            
            
            
        
