import math

import numpy as np
from src.train.scheduler.scheduler import Scheduler


class scheduler_0(Scheduler):
    def __init__(self, idx, base_lr, final_lr, epochs, warmup_epochs,
                 dataloader_len):
        super().__init__()
        warmup_lr_schedule = np.linspace(0, base_lr,
                                         int(dataloader_len * warmup_epochs))
        iters = np.arange(dataloader_len * (epochs - warmup_epochs))
        cosine_lr_schedule = np.array([
            final_lr + 0.5 * (base_lr - final_lr) *
            (1 + math.cos(math.pi * t / (dataloader_len *
                                         (epochs - warmup_epochs))))
            for t in iters
        ])
        self.lr_schedule = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))

    def get_lr_for_iter(self, iteration):
        return self.lr_schedule[iteration]
