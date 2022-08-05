

import torch.nn as nn
import torch

class TrajectoryCenterCrop(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x):
        print(x.shape)
        x = x[:,:,0:80, 20:100]
        print(x.shape)
        assert False
        return


