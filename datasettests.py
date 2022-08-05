"""
temporary file
"""
import numpy as np
import torch

from src.dataset.dataset import dataset
import matplotlib.pyplot as plt
import time
import json
import numpy
import seaborn as sns
import random
from tqdm import tqdm, trange
def pprint(a):
    for k in a.keys():
        el = a[k]
        if isinstance(el, torch.Tensor):
            print(f"tensor {k} shape {el.shape}")
        if isinstance(el, list):
            print(f"list {k} shape {len(el)}")

representation_type = 'image'
dataset_name = "lyft"

cfg = [{'name':dataset_name,'augmentation_type':{"no":0},'mode':'train','bbox_meter': [60, 60],'bbox_pixel':[120,120],'center_meter':[20.0,30.0],\
           'hist_seq_first':0,'hist_seq_last':19,'pred_seq_last':49,'representation_type':'image_multichannel_vector'}]

d= dataset(**cfg[0])
print(len(d))
for a in [d[0], d[1]]:
    pprint(a)
    print()
