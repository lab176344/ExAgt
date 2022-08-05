from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
from glob import glob
from pathlib import Path
import torch
import numpy
import cv2
import os
import random
from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range

'''
Input array structure

ego/obj[0,:] = X
ego/obj[1,:] = Y
ego/obj[2,:] = TIMESTAMP
ego/obj[3,:] = psi (angle velocity)
ego/obj[4,:] = v
ego/obj[5,:] = a_lon
ego/obj[6,:] = a_lat
ego/obj[7,:] = time_idx_in_scenario_frame !!starting with 1!!
ego/obj[8,:] = circle_or_rectangle

ego[7,:] = [ 1 2 3 4 5 6 7 8 9 10]

obj[7,:] = [ 4 5 6] --> [0 0 0 4 5 6 0 0 0 0]
ego[7,:] = [ 8 9 10]

TODO:
introduce hist/pred split to all loaders
    HINT: obj_inner = obj[:, (time_idx_temp >= self.seq_first) & (time_idx_temp <= self.seq_last)]
interpolation?

'''

augmentation_types = ["no","connectivity","fieldofview","range"]
representation_types = ["vector","image_vector","image_multichannel_vector","image","image_bound","graph","trajectory","image_vector_merge_recon"]

class dataset(Dataset):
    # IMPORT LOADER METHODS HERE
    from .loaders._vector import _load_sample_vector
    from .loaders._image_vector import _load_sample_image_vector
    from .loaders._image import _load_sample_image
    from .loaders._image_bound import _load_sample_image_bound
    from .loaders._graph import _load_sample_graph
    from .loaders._image_vector_merge_recon import _load_sample_image_vector_merge_recon
    from .loaders._image_multichannel_vector import _load_sample_image_multichannel_vector
    
    def __init__(self, name='argoverse', augmentation_type=None, augmentation_meta=None, representation_type='image', orientation='plain',
                 mode='train', only_edges=False, bbox_meter=[200.0, 200.0], bbox_pixel=[100, 100], center_meter=None,
                 hist_seq_first=0, hist_seq_last=49, pred_seq_first=None, pred_seq_last=None):
        super().__init__()
        if augmentation_type is None:
            augmentation_type = {}
        self.name = name
        path = (Path(__file__).parent.parent.parent).joinpath('data').joinpath(self.name)
        self.dir = str(path)
        assert representation_type in representation_types, 'representation keyword unknown'
        self.representation_type = representation_type
        self.orientation = orientation
        self.mode = mode
        self.only_edges = only_edges
        #TODO
        #assert augmentation_type in augmentation_types, 'augmentation keyword unknown'
        self.augmentation_type = augmentation_type
        self.augmentation_meta = augmentation_meta

        self.bbox_meter = bbox_meter
        self.bbox_pixel = bbox_pixel
        self.center = center_meter
        self.hist_seq_first = hist_seq_first
        self.hist_seq_last = hist_seq_last
        self.hist_seq_len = self.hist_seq_last - self.hist_seq_first + 1
        
        self.pred_seq_first = pred_seq_first
        if not pred_seq_first:
            self.pred_seq_first = self.hist_seq_last +1 
        self.pred_seq_last  = pred_seq_last
        if not pred_seq_last:
            self.pred_seq_len = None
        else:
            self.pred_seq_len   = self.pred_seq_last - self.pred_seq_first + 1

        self.only_edges = only_edges
        self.min_y_meter = - self.center[1]
        self.min_x_meter = - self.center[0]
        self.max_y_meter = self.min_y_meter + self.bbox_meter[1]
        self.max_x_meter = self.min_x_meter + self.bbox_meter[0]
        self.rect_width = 1.5 # compact car
        self.rect_length = 4 # compact car
        self.square_size = 0.5 

        # Check if there is any train/test/val split
        result = [f.name for f in path.rglob("*")]
        if 'train' in result and 'test' in result:
            train_test_split_valid = True
        else:
            print('Please generate train/test split beforehand')
            exit()

        self.files_augmented = sorted(
                glob(self.dir + "/" + self.mode + "/augmentation/*.mat"))
        self.files_base = sorted(glob(self.dir + "/" + self.mode + "/base/*.mat"))

        # finetuning remove this to keep it general
        if False and mode == "train":
            # alternatively, there is an idx1.npy file containing the indices for the 1% fine tuning
            file ="./idx10.npy"
            idx_subset = numpy.load(file)
            print(f"Warning only using a subset of the data {len(idx_subset)}")

            self.files_augmented = [self.files_augmented[idx] for idx in idx_subset]
            self.files_base = [self.files_base[idx] for idx in idx_subset]
            print(f"reduced dataset contains {len(self.files_augmented)} samples")



        # assert len(self.files_base) == len(self.files_augmented)
        self.len = len(self.files_base)
        self.data = [None] * self.len

        # Load the data into a list of dictionaries
        if self.representation_type in ['vector', 'image_bound']:
            # Load the maps--------------------------
            map_files = glob(self.dir + "\\map\\*.mat")
            self.map = {}
            ways_bbox = {}
            for map_file in map_files:
                map_temp = loadmat(map_file)
                map_ID = os.path.basename(map_file)[:-4]

                way_ids = map_temp['way_id'][0]
                way_inside = map_temp['way_bound_in'][0]
                way_outside = map_temp['way_bound_out'][0]

                self.map[map_ID] = (way_ids, way_inside, way_outside, map_temp['way_bbox'])
                ways_bbox[map_ID] = map_temp['way_bbox']

    def __getitem__(self, index):
        if self.representation_type == "image":
            return self._load_sample_image(self._get_file(index))
        if self.representation_type == "image_vector":
            return self._load_sample_image_vector(self._get_file(index))
        if self.representation_type == "image_multichannel_vector":
            return self._load_sample_image_multichannel_vector(self._get_file(index))
        if self.representation_type == "vector":
            return self._load_sample_vector(self._get_file(index))
        if self.representation_type == "image_bound":
            return self._load_sample_image_bound(self._get_file(index))
        if self.representation_type == "graph":
            return self._load_sample_graph(self._get_file(index))
        if self.representation_type == "trajectory":
            return self._load_sample_trajectory(self._get_file(index))
        if self.representation_type == "image_vector_merge_recon":
            return self._load_sample_image_vector_merge_recon(self._get_file(index))

    def __len__(self):
        return self.len

    def _get_file(self, idx):
        if "connectivity" in self.augmentation_type.keys() and random.random() < \
                self.augmentation_type["connectivity"]:
            return self.files_augmented[idx]
        return self.files_base[idx]

    def __str__(self):
        return "Loaded {} dataset: {} total samples: {}".format(self.mode,
                                                                self.name,
                                                                len(self))

    def _get_description(self):
        # TODO
        description = {'name': self.name}
        return description
