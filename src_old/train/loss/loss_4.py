import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from src.train.loss.loss import loss


class loss_4(loss):
    def __init__(self, idx=4, name="Vicr", description="Simple Siamese Loss", input_="z1,z2", output="loss",
                 version='simplified'):
        super().__init__(idx, name, description, input_, output)
        self.version = version
        
    def forward(self, p, z): # negative cosine similarity
        if  self.version=='fancy':
            z = z.detach() # stop gradient
            p = F.normalize(p, dim=1) # l2-normalize 
            z = F.normalize(z, dim=1) # l2-normalize 
            return -(p*z).sum(dim=1).mean()

        elif self.version=='simplified':# same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception

   