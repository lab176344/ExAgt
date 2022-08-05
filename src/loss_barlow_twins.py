import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__(self,
                lambda_param = 0.005,
                ) -> None:
        self.lambda_param = lambda_param

    def forward(self,out_1 = None, out_2 = None, epoch = 0, device=None):
        batch_size = out_1.size(0)
        D = out_1.size(1)
        # cross-correlation matrix
        c = torch.mm(out_1.T, out_2) / batch_size
        # loss
        c_diff = (c - torch.eye(D,device=device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
    
        return loss
        