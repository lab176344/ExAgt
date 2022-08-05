# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:04:23 2021

@author: neumeierm
"""

from torch.utils.data.dataloader import default_collate
import torchnet as tnt


class dataloader_1(object):
    def __init__(self,
                 idx=1,
                 dataset=None,
                 batch_size=None,
                 epochs = None,
                 num_workers=None,
                 shuffle=None,
                 transformation=None,
                 representation='graph',
                 name=None,
                 description='Marion Trajectory Prediction',
                 test=False,
                 num_gpus=1):

        self.idx = idx
        self.dataset = dataset
        self.name = name
        self.size = None
        self.n_samples = None
        self.image = 0
        self.image_vector = 0
        self.vector = 0
        self.graph = 0
        self.description = description
        self.test = test
        self.epoch_size = epochs if epochs is not None else len(dataset)
        self.epochs = epochs
        self.num_gpus = num_gpus

        if representation is not 'graph':
            print('Please select graph as representation')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.tranformation = transformation

    def get_iterator(self,epoch, rank):
        def _load_function(idx):
            F_X = self.dataset[idx]['F_X']
            F_Y = self.dataset[idx]['F_Y']
            F_VX = self.dataset[idx]['F_VX']
            F_VY = self.dataset[idx]['F_VY']
            XY_pred = self.dataset[idx]['XY_pred']
            return F_X, F_Y, F_VX, F_VY, XY_pred

        def _collate_fun(batch):
            batch = default_collate(batch)
            assert(len(batch) == 5)
            return batch
            ## TODO : add multiple gpu support
        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch, rank=0):
        return self.get_iterator(epoch, rank)

