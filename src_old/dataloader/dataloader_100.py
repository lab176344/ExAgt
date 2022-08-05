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
        self.r2int = 10
        self.grid_chosen = grid_chosen
        self.epoch_size = epoch_size if epoch_size is not None else len(
            dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers,
        self.epochs = epochs
        self.dataset = dataset
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
        print('test')


    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        # Get acnhor
        a = self.dataset[idx]
        possible_pns = np.setdiff1d(self.infra_container[self.label_infra[idx]][0],idx)
        possible_pps = np.setdiff1d(self.route_container[self.label_route[idx]][0],idx)
        if possible_pns.size == 0:
            pn = self.dataset[idx]
        else:
            idx_neighbor_idx = torch.randint(possible_pns.size,(1,))
            idx_neighbor = possible_pns[np.squeeze(idx_neighbor_idx.numpy()).astype(int)]
            pn = self.dataset[idx_neighbor]

        if possible_pps.size == 0:
            pp = self.dataset[idx]
        else:
            # TODO ensure to not be the same as pn?
            idx_neighbor_idx = torch.randint(possible_pps.size,(1,))
            idx_neighbor = possible_pps[np.squeeze(idx_neighbor_idx.numpy()).astype(int)]
            pp = self.dataset[idx_neighbor]

        a = a['image']
        pp = pp['image']
        pn = pn['image']
        
        return a, pp, pn

    def _collate_fun(self, batch):
        self.rand_seed1 += self.r1int
        self.rand_seed2 += self.r2int
        batch = default_collate(batch)
        assert (len(batch) == 4)
        return batch

    def get_iterator(self, epoch, gpu_idx):
        self.rand_seed1 = epoch
        self.rand_seed2 = epoch + 1

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