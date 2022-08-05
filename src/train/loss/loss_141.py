import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F

class loss_141(loss):
    def __init__(self,
                idx = 141,
                name = 'FDE',
                description = 'Final Displacement Error: The RMSE between the ground truth and the predicted trajectory position at the last time frame.',
                input_ = '',
                output = '1D',
                pred_ = 50,
                ) -> None:
        super().__init__(idx,name,description,input_,output)
        self.pred_frames = pred_
        
    def forward(self, out_1 = None, out_2 = None):
        fde = F.mse_loss(out_1[-1], out_2[-1])
        return fde


