import random
from scipy.io import loadmat
from glob import glob
import os

import numpy as np
import torch
import torchnet as tnt
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms


class dataloader_100(object):
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
        
        # Load the class file
        path = glob(self.dataset.dir + "/" + self.dataset.mode + "/infra_route_sim.mat")[0]
        map_temp = loadmat(path)
        scenario_id = map_temp['scenario_id'][0]
        scenario_id = [ele[0] for ele in scenario_id]
        scenario_id_sorting = sorted(range(len(scenario_id)),key=scenario_id.__getitem__)
        scenario_id_sorted = [ scenario_id[val] for val in scenario_id_sorting]
        scenario_id_from_files = [ os.path.basename(path)[:-4] for path in self.dataset.files]
        if not scenario_id_from_files == scenario_id_sorted:
            print('scenario_ids do not match. Labels can not be assigned')
            quit()
        scenario_id_sorting = np.array(scenario_id_sorting)
        self.label_infra = map_temp['label_infra'][0][scenario_id_sorting]-1
        self.label_route = map_temp['label_route'][0][scenario_id_sorting]-1

        infra_un = np.unique(self.label_infra)
        infra_container = [None]*len(infra_un)
        for infra_class in infra_un:
            infra_container[infra_class] =  np.where(self.label_infra==infra_class)
        self.infra_container = infra_container

        route_un = np.unique(self.label_route)
        route_container = [None]*len(route_un)
        for route_class in route_un:
            route_container[route_class] =  np.where(self.label_route==route_class)
        self.route_container = route_container


    def _load_function(self, idx):
        #torch.manual_seed(self.rand_seed1)
        idx = idx % len(self.dataset)
        # Get acnhor
        a = self.dataset[idx]
        possible_pns = np.setdiff1d(self.infra_container[self.label_infra[idx]][0],self.route_container[self.label_route[idx]][0])
        possible_pps = np.setdiff1d(self.route_container[self.label_route[idx]][0],idx)
        if possible_pns.size == 0:
            idx_neighbor = idx
        else:
            idx_neighbor_idx = torch.randint(possible_pns.size,(1,))
            idx_neighbor = possible_pns[np.squeeze(idx_neighbor_idx.numpy()).astype(int)]
        
        pn = self.dataset[idx_neighbor]
        #pn_idx = idx_neighbor

        if possible_pps.size == 0:
            idx_neighbor = idx
        else:
            idx_neighbor_idx = torch.randint(possible_pps.size,(1,))
            idx_neighbor = possible_pps[np.squeeze(idx_neighbor_idx.numpy()).astype(int)]
        
        pp = self.dataset[idx_neighbor]
        #pp_idx = idx_neighbor

        # Load Targets here
        return a, pp, pn#, idx, pp_idx, pn_idx

    def _collate_fun(self, batch):
        self.rand_seed1 += self.r1int
        lens_seq_ego_a = []
        stacked_ego_a = []

        lens_seq_ego_pp = []
        stacked_ego_pp = []

        lens_seq_ego_pn = []
        stacked_ego_pn = []

        
        stacked_image_a = torch.zeros((len(batch),batch[0][0]['image'].shape[0],batch[0][0]['image'].shape[1],batch[0][0]['image'].shape[2]))
        stacked_image_a_target = torch.zeros((len(batch),batch[0][0]['image_merged'].shape[0],batch[0][0]['image_merged'].shape[1],batch[0][0]['image_merged'].shape[2]))
        stacked_image_pp = torch.zeros((len(batch),batch[0][0]['image'].shape[0],batch[0][0]['image'].shape[1],batch[0][0]['image'].shape[2]))
        stacked_image_pn = torch.zeros((len(batch),batch[0][0]['image'].shape[0],batch[0][0]['image'].shape[1],batch[0][0]['image'].shape[2]))
        for idx,elems in enumerate(batch):
            stacked_image_a[idx,:,:,:] = elems[0]['image']
            stacked_image_a_target[idx,:,:,:] = elems[0]['image_merged']
            stacked_ego_a.append(torch.from_numpy(elems[0]['traj_ego']).t().float())
            lens_seq_ego_a.append(elems[0]['traj_ego'].shape[1])

            stacked_image_pp[idx,:,:,:] = elems[1]['image']
            stacked_ego_pp.append(torch.from_numpy(elems[1]['traj_ego']).t().float())
            lens_seq_ego_pp.append(elems[1]['traj_ego'].shape[1])

            stacked_image_pn[idx,:,:,:] = elems[2]['image']
            stacked_ego_pn.append(torch.from_numpy(elems[2]['traj_ego']).t().float())
            lens_seq_ego_pn.append(elems[2]['traj_ego'].shape[1])

        padded_ego_a = torch.nn.utils.rnn.pad_sequence(stacked_ego_a, batch_first=True)
        padded_ego_pp = torch.nn.utils.rnn.pad_sequence(stacked_ego_pp, batch_first=True)
        padded_ego_pn = torch.nn.utils.rnn.pad_sequence(stacked_ego_pn, batch_first=True)
        out =   (stacked_image_a,padded_ego_a,lens_seq_ego_a, stacked_image_a_target),\
                (stacked_image_pp,padded_ego_pp,lens_seq_ego_pp),\
                (stacked_image_pn,padded_ego_pn,lens_seq_ego_pn)
        return  out

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