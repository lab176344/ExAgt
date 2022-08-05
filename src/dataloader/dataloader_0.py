import math
import random
import numpy as np
import torch
import torchnet as tnt
from torch.utils.data.dataloader import default_collate


class dataloader_0(object):
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
        uniform_sample=False,
        grid_chosen=None,
        name='Cross View Prediction Dataloader',
        description='Dataloader which gives two augumented version of the scenario with the target labels if available'
    ):

        self.transform = transformation
        self.r1int = 3 # some random seet to start with
        self.r2int = 10 # some random seet to start with different seeds for the second augumentation
        self.grid_chosen = grid_chosen
        self.epoch_size = epoch_size if epoch_size is not None else len(
            dataset[0])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers,
        self.epochs = epochs
        self.dataset = dataset
        self.uniform_sample = uniform_sample
        self.num_gpus = num_gpus


    def _load_function(self, idx):
        idx = idx % len(self.dataset[0])
        datasetwoAug = self.dataset[0]
        datasetwAug = self.dataset[1]
        x1 = datasetwoAug[idx]['image']
        x2 = datasetwAug[idx]['image']
        y = datasetwoAug[idx]['label']
        x1 = x1.type(torch.uint8)
        x2 = x2.type(torch.uint8)

        rotated_imgs1, rotated_imgs2 = self.generate_random_sequence(
            x1, x2, len(self.grid_chosen), self.transform,
            self.rand_seed1, self.rand_seed2)
        return rotated_imgs1, rotated_imgs2, y, idx

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
        self.data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=self._collate_fun,
                                           num_workers=self.num_workers[0],
                                           sampler=sampler)
        return self.data_loader

    def __call__(self, epoch=0, rank=0):
        return self.get_iterator(epoch, rank)

    def __len__(self):
        return math.ceil((len(self.dataset[0]) / self.batch_size) / self.num_gpus)

    def generate_random_sequence(self, scenario1: torch.Tensor,
                                 scenario2: torch.Tensor, temporal_len_grids,
                                 custom_transforms, rseed1,
                                 rseed2):
        """[Generates random sequences based on the OGs and the number of OGs chosen]

        Args:
            scenario ([array (L,W,T)]): [The scenario as stacked occupancy grids]
            stuff ([list]): [The correct order and the OGs to be chosen for the SSL]
            custom_transforms ([type]): [transforms]

        Returns:
            [tuple]: [description]
        """

        custom_transforms1 = custom_transforms[0]
        custom_transforms2 = custom_transforms[1]
        USE_SAME_SAMPLING = False

        trans_clip1 = []
        trans_clip2 = []
        random.seed(rseed1)
        if not self.uniform_sample:
            id_1 = random.sample(range(scenario1.shape[1]), temporal_len_grids)
        else:
            id_1 = np.array(self.grid_chosen)

        id_1_sort = np.sort(id_1)
        for k in id_1_sort:
            random.seed(rseed1)
            torch.manual_seed(rseed1)
            frame = scenario1[0, k, :, :]
            if custom_transforms1 is not None:
                frame = custom_transforms1(frame.view(1, 1,
                                                     *frame.shape)).squeeze(0)
            else:
                frame = frame.unsqueeze(0)
            trans_clip1.append(frame)

        trans_clip1 = torch.cat(trans_clip1).permute([0, 2, 1]).unsqueeze(0)

        random.seed(rseed2)

        if not self.uniform_sample:
            id_2 = random.sample(range(scenario1.shape[1]), temporal_len_grids)
        else:
            id_2 = np.array(self.grid_chosen)
        id_2_sort = np.sort(id_2)

        if USE_SAME_SAMPLING:
            id_2_sort = id_1_sort

        for k in id_2_sort:
            random.seed(rseed2)
            torch.manual_seed(rseed2)
            frame = scenario2[0, k, :, :]
            if custom_transforms2 is not None:
                frame = custom_transforms2(frame.view(1, 1,
                                                     *frame.shape)).squeeze(0)
            else:
                frame = frame.unsqueeze(0)
            trans_clip2.append(frame)
        trans_clip2 = torch.cat(trans_clip2).permute([0, 2, 1]).unsqueeze(0)

        return trans_clip1, trans_clip2
