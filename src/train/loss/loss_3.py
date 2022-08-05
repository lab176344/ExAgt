import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from src.train.loss.loss import loss


class loss_3(loss):
    def __init__(self, idx=4, name="Vicr", description="Variance Invariance Convariance loss", input_="z1,z2", output="loss",
                 wtSim=0.25, wtVar=0.25, wtCov=0.25):

        super().__init__(idx, name, description, input_, output)
        self.wtSim = wtSim
        self.wtVar = wtVar
        self.wtCov = wtCov
    def invarianceLoss(self,z1=None, z2=None):
        return F.mse_loss(z1, z2)
    
    def varianceLoss(self,z1=None, z2=None):
        eps = 1e-04 # From the paper
        stdZ1 = torch.sqrt(z1.var(dim=0) + eps)
        stdZ2 = torch.sqrt(z2.var(dim=0) + eps)
        stdLoss = torch.mean(F.relu(1 - stdZ1))
        stdLoss = stdLoss  + torch.mean(F.relu(1 - stdZ2))
        return stdLoss
    def covarianceLoss(self,z1=None, z2=None):
        bachSize, D = z1.size()

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        covZ1 = (z1.T @ z1) / (bachSize - 1)
        covZ2 = (z2.T @ z2) / (bachSize - 1)
        diag = torch.eye(D, device=z1.device)
        covLoss = covZ1[~diag.bool()].pow_(2).sum() / D + covZ2[~diag.bool()].pow_(2).sum() / D
        return covLoss

    def forward(self,z1=None, z2=None,device=None):
        invarianceLoss = self.invarianceLoss(z1, z2)
        varianceLoss = self.varianceLoss(z1, z2)
        convarianceLoss = self.covarianceLoss(z1, z2)      
        loss = invarianceLoss * self.wtSim + varianceLoss * self.wtVar + convarianceLoss * self.wtCov
        return loss

   
