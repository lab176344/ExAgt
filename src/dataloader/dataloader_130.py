import random
from scipy.io import loadmat
from glob import glob
import os

import numpy as np
import torch
import torchnet as tnt
import math



class dataloader_130(object):
    def __init__(
        self,
        idx=130,
        dataset=None,
        batch_size=None,
        epochs=None,
        num_workers=0,
        num_gpus=1,
        shuffle=None,
        epoch_size=None,
        transformation=None,
        transformation3D=None,
        representation='TODO',
        test=False,
        grid_chosen=None,
        name='TODO',
        description='TODO'
    ):
        self.dataset = dataset[0]
        self.epoch_size = epoch_size if epoch_size is not None else len(
            self.dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers,
        self.epochs = epochs
        
        self.test = test
        self.num_gpus = num_gpus

    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        sample = self.dataset[idx]
        return sample

    def _collate_fun(self, batch):
        hist_objs           = []
        hist_objs_seq_len   = []
        hist_obj_lens       = []
      
        pred_objs           = []
        pred_objs_seq_len   = []
        pred_obj_lens       = []        
        multi_channel_images = []
        
        for elems in batch:
            multi_channel_images.append(elems['image'])
            
            hist_objs.append(           elems['traj_hist_obj'])
            hist_objs_seq_len.append(   elems['traj_hist_obj_seq_lens']) 
            hist_obj_lens.append(       elems['traj_hist_obj'].shape[0])
            pred_objs.append(           elems['traj_pred_obj'])
            pred_objs_seq_len.append(   elems['traj_pred_obj_seq_lens']) 
            pred_obj_lens.append(       elems['traj_pred_obj'].shape[0])
            
        hist_objs   = torch.nn.utils.rnn.pad_sequence(hist_objs, batch_first=True)
        pred_objs   = torch.nn.utils.rnn.pad_sequence(pred_objs, batch_first=True)
        
        
        hist_objs_seq_len = [torch.Tensor(ele) for ele in hist_objs_seq_len]
        hist_objs_seq_len   = torch.nn.utils.rnn.pad_sequence(hist_objs_seq_len, batch_first=True)
        pred_objs_seq_len = [torch.Tensor(ele) for ele in pred_objs_seq_len]
        pred_objs_seq_len   = torch.nn.utils.rnn.pad_sequence(pred_objs_seq_len, batch_first=True)
        multi_channel_images = torch.stack(multi_channel_images, dim=0)

        out_dict = {'images': multi_channel_images,'hist_objs': hist_objs,'hist_obj_lens': torch.Tensor((hist_obj_lens)), 'hist_objs_seq_len': torch.Tensor((hist_objs_seq_len)),
                    'pred_objs': pred_objs,'pred_obj_lens': torch.Tensor((pred_obj_lens)), 'pred_objs_seq_len': torch.Tensor((pred_objs_seq_len)),
                    }
        return  out_dict
        

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
                                           sampler=sampler)
        return data_loader
    
    def __call__(self, epoch=0, rank=0):
        return self.get_iterator(epoch, rank)
    
    def __len__(self):
        return math.ceil((len(self.dataset) / self.batch_size) / self.num_gpus)