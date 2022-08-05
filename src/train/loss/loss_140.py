import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F

class loss_140(loss):
    def __init__(self,
                idx = 140,
                name = 'ADE',
                description = 'Average Displacement Error: The average of the RMSE between the ground truth and the predicted trajectory position at every time frame for the entire duration of the prediction hoirzon.',
                input_ = '',
                output = '1D',
                ) -> None:
        super().__init__(idx,name,description,input_,output)
        
    def forward(self, out_1 = None, out_2 = None):
        ade = F.mse_loss(out_1, out_2, reduction = 'mean')
        return ade