import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F

class loss_120(loss):
    def __init__(self,
                idx = 120,
                name = 'TODO',
                description = 'TODO',
                input_ = 'TODO',
                output = 'TODO',
                ) -> None:
        super().__init__(idx,name,description,input_,output)
    def forward(self, pred = None, gTruth = None):
        
        return F.mse_loss(pred, gTruth)
