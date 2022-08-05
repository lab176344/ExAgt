import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F

class loss_1(loss):
    def __init__(self,
                idx = 1,
                name = 'MSE Loss',
                description = 'Mean square error for 2D trajectory prediction',
                input_ = '',
                output = '1D',
                ) -> None:
        super().__init__(idx,name,description,input_,output)
    def forward(self, out_1 = None, out_2 = None, PRED_FRAMES = 125):
        RCL_y = F.mse_loss(out_1[PRED_FRAMES:], out_2[PRED_FRAMES:], reduction = 'sum')
        RCL_x = F.mse_loss(out_1[:PRED_FRAMES], out_2[:PRED_FRAMES], reduction = 'mean')
        RCL = RCL_y + RCL_x
        return RCL
