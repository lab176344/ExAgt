import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F

class loss_122(loss):
    def __init__(self,
                idx = 122,
                name = 'TODO',
                description = 'TODO',
                input_ = 'TODO',
                output = 'TODO',
                ) -> None:
        super().__init__(idx,name,description,input_,output)
    def forward(self, pred = None, gTruth = None):
        
        return F.huber_loss(pred, gTruth)
