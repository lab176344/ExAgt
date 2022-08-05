from __future__ import print_function
import torch
from torch.utils.data import TensorDataset
from src.dataloader.dataloader_0 import dataloader_0
from torch.utils.data.dataset import Dataset
import numpy as np


def get_datasets(n_train=10, n_valid=1024,
                 input_shape=[1, 4, 200, 200], target_shape=[],
                 n_classes=None):
    """Construct and return random number datasets"""
    data = [None]*n_train
    y = [None]*n_train
    for idx in range(n_train):
        image = torch.randn([n_train] + input_shape)
        dictionary_temp = {'image': image}
        data[idx] = dictionary_temp
        y[idx] = torch.randn([n_train] + target_shape)
    return data, y


class MyDataset(Dataset):
    def __init__(self):
        data, y = get_datasets()
        self.x = data
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x

    def __len__(self):
        return len(self.x)


test_data = MyDataset()
loader = dataloader_0(dataset=test_data,
                    batch_size=2,
                    epochs = 10,
                    num_workers=0,
                    shuffle=False,
                    transformation=None,
                    transformation3D=None,
                    test=False,
                    grid_chosen=[0, 3, 6, 9],
                    representation='image')
descriptio = loader._get_description()
print(descriptio)
