import random
from scipy.io import loadmat
from glob import glob
import os

import numpy as np
import torch
import torchnet as tnt
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms


class dataloader_101(object):
    def __init__(
        self,
        idx=0,
        dataset=None,
        batch_size=None,
        epochs=None,
        num_workers=0,
        num_gpus=1,
        shuffle=None,
        epoch_size=None,
        transformation=None,
        transformation3D=None,
        representation='image',
        test=False,
        grid_chosen=None,
        name='Triplet Dataloader',
        description='TODO'
    ):
        # super().__init__(idx,
        #                  dataset,
        #                  batch_size,
        #                  epochs,
        #                  num_workers,
        #                  shuffle,
        #                  transformation,
        #                  representation,
        #                  name,
        #                  description,
        #                  test)
        self.transform = transformation
        self.tranformation3D = transformation3D
        self.r1int = 3
        self.grid_chosen = grid_chosen
        self.epoch_size = epoch_size if epoch_size is not None else len(
            dataset[0])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers,
        self.epochs = epochs
        self.dataset = dataset[0]
        self.test = test
        self.num_gpus = num_gpus
        
        # Load classes per file and prepare class dicts
        


    def _load_function(self, idx):
        #torch.manual_seed(self.rand_seed1)
        idx = idx % len(self.dataset)
        sample = self.dataset[idx]
        return sample

    def _collate_fun(self, batch):
        self.rand_seed1 += self.r1int
        lens_seq_ego = []
        stacked_ego = []

        stacked_image = torch.zeros((len(batch),batch[0]['image'].shape[0],batch[0]['image'].shape[1],batch[0]['image'].shape[2]))
        stacked_image_target = torch.zeros((len(batch),batch[0]['image_merged'].shape[0],batch[0]['image_merged'].shape[1],batch[0]['image_merged'].shape[2]))
        stacked_label = torch.zeros((len(batch),))

        for idx,elems in enumerate(batch):
            stacked_image[idx,:,:,:] = elems['image']
            stacked_image_target[idx,:,:,:] = elems['image_merged']
            stacked_label[idx] = elems['label'][0]
            stacked_ego.append(torch.from_numpy(elems['traj_ego']).t().float())
            lens_seq_ego.append(elems['traj_ego'].shape[1])

        padded_ego = torch.nn.utils.rnn.pad_sequence(stacked_ego, batch_first=True)
        return stacked_image, padded_ego, lens_seq_ego, stacked_image_target, stacked_label

    def get_iterator(self, epoch, gpu_idx):
        self.rand_seed1 = epoch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=self._load_function)

        sampler = torch.utils.data.distributed.DistributedSampler(
            tnt_dataset,
            num_replicas=self.num_gpus,
            shuffle=self.shuffle,
            rank=gpu_idx)
        sampler.set_epoch(epoch)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=self._collate_fun,
                                           num_workers=self.num_workers[0],
                                           sampler=sampler,)
        return data_loader

    def __call__(self, epoch=0, rank=0):
        return self.get_iterator(epoch, rank)