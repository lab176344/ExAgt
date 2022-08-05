import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

class optimizer_1(optim.Adam):
    def __init__(self,
                params = Parameter(torch.randn(2, 2, requires_grad=True)),
                lr = 0.001, 
                weight_decay = 0, 
                betas = (0.9, 0.999),
                eps=1e-08,
                amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, **defaults)