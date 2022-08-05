import random
from scipy.io import loadmat
from glob import glob
import os

import numpy as np
import torch
import torchnet as tnt
import math



class dataloader_120(object):
    def __init__(
        self,
        idx=120,
        dataset=None,
        batch_size=None,
        epochs=None,
        num_workers=0,
        num_gpus=1,
        shuffle=False,
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
        pred_objsx          = []
        pred_objsy          = []
        pred_objst          = []
        pred_objs_seq_len   = []
        pred_obj_lens       = []        
        multi_channel_images = torch.zeros(len(batch), 1, 240, 240)

        obj_decoder_in = []
        
        for idx,elems in enumerate(batch):
            multi_channel_images[idx,:,:,:] = elems['image']
            hist_objs.append(           elems['traj_hist_obj'])
            hist_objs_seq_len        += elems['traj_hist_obj_seq_lens']
            hist_obj_lens.append(       elems['traj_hist_obj'].shape[0])
            pred_objsx.append(           elems['traj_pred_obj'][:,:,0].unsqueeze(2))
            pred_objsy.append(           elems['traj_pred_obj'][:,:,1].unsqueeze(2))
            pred_objst.append(           elems['traj_pred_obj'][:,:,2].unsqueeze(2))
            pred_objs_seq_len        += elems['traj_pred_obj_seq_lens'] 
            pred_obj_lens.append(       elems['traj_pred_obj'].shape[0])
            obj_decoder_in.append(      elems['obj_decoder_in'])
        # padding to handle different number of objects
        hist_objs               = torch.nn.utils.rnn.pad_sequence(hist_objs, batch_first=True) # batch_size x max_obj x max_obj_seq_len x feature_dim
        pred_objsx              = torch.nn.utils.rnn.pad_sequence(pred_objsx, batch_first=True,padding_value=torch.nan) # batch_size x max_obj x max_obj_seq_len x feature_dim
        pred_objsy              = torch.nn.utils.rnn.pad_sequence(pred_objsy, batch_first=True,padding_value=torch.nan) # batch_size x max_obj x max_obj_seq_len x feature_dim
        pred_objst              = torch.nn.utils.rnn.pad_sequence(pred_objst, batch_first=True,padding_value=torch.nan) # batch_size x max_obj x max_obj_seq_len x feature_dim
        obj_decoder_in          = torch.nn.utils.rnn.pad_sequence(obj_decoder_in, batch_first=True)
        hist_objs_seq_len       = [ele for ele in hist_objs_seq_len]
        hist_object_lengths_sum = torch.cumsum(torch.Tensor(hist_obj_lens),0).int()
        hist_object_lengths_sum = torch.cat([torch.Tensor([0]),hist_object_lengths_sum]).int() # num of objects per scene , cumsum of the number of objects in the scene

        pred_objs_seq_len       = [ele for ele in pred_objs_seq_len]
        #multi_channel_images    = torch.stack(multi_channel_images, dim=0)
        pres_object_lengths_sum = torch.cumsum(torch.Tensor(pred_obj_lens),0).int()
        pres_object_lengths_sum = torch.cat([torch.Tensor([0]),pres_object_lengths_sum]).int()

        out_dict = {'images': multi_channel_images,'hist_objs': hist_objs,'hist_obj_lens': hist_obj_lens, 'hist_objs_seq_len': hist_objs_seq_len,
                    'pred_objsx': pred_objsx, 'pred_objsy': pred_objsy, 'pred_objst':pred_objst, 'pred_obj_lens': pred_obj_lens, 'pred_objs_seq_len': pred_objs_seq_len, 'obj_decoder_in': obj_decoder_in,
                    'hist_object_lengths_sum': hist_object_lengths_sum, 'pres_object_lengths_sum': pres_object_lengths_sum}
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
                                           sampler=sampler,pin_memory=True)
        return data_loader
    
    def __call__(self, epoch=0, rank=0):
        return self.get_iterator(epoch, rank)
    
    def __len__(self):
        return math.ceil((len(self.dataset) / self.batch_size) / self.num_gpus)