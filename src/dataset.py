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
import numpy as np

'''
Input array structure

ego/obj[0,:] = X
ego/obj[1,:] = Y
ego/obj[2,:] = TIMESTAMP
ego/obj[3,:] = psi (angle velocity)
ego/obj[7,:] = time_idx_in_scenario_frame !!starting with 1!!
ego/obj[8,:] = circle_or_rectangle

'''

class dataset(Dataset):
    # IMPORT LOADER METHODS HERE
    def __init__(self, name='argoverse', augmentation_type=None, augmentation_meta=None, orientation='plain',
                 mode='train', bbox_meter=[200.0, 200.0], bbox_pixel=[100, 100], center_meter=None,
                 seq_first=0, seq_last=49):
        super().__init__()
        if augmentation_type is None:
            augmentation_type = {}
        self.name = name
        path = (Path(__file__).parent.parent).joinpath('data').joinpath(self.name)
        self.dir = str(path)
        self.orientation = orientation
        self.mode = mode
        self.augmentation_type = augmentation_type
        self.augmentation_meta = augmentation_meta

        self.bbox_meter = bbox_meter
        self.bbox_pixel = bbox_pixel
        self.center = center_meter
        self.seq_first = seq_first
        self.seq_last = seq_last
        self.seq_len = self.seq_last - self.seq_first + 1

        self.min_y_meter = - self.center[1]
        self.min_x_meter = - self.center[0]
        self.max_y_meter = self.min_y_meter + self.bbox_meter[1]
        self.max_x_meter = self.min_x_meter + self.bbox_meter[0]
        self.rect_width = 1.5 # compact car
        self.rect_length = 4 # compact car
        self.square_size = 0.5 

        self.files_augmented = sorted(
                glob(self.dir + "/" + self.mode + "/augmentation/*.mat"))
        self.files_base = sorted(glob(self.dir + "/" + self.mode + "/base/*.mat"))

        self.len = len(self.files_base)
        self.data = [None] * self.len

    def _load_sample_image(self, filename):
        # IMAGE=================================================================================
        mat_temp = loadmat(filename, variable_names=["image", "orient", "image_size_meter", "ego", "obj_list","label_split"],
                            verify_compressed_data_integrity=False)
        bbox_meter_in = mat_temp['image_size_meter'].astype(numpy.int16)[0]
        orientation = float(-mat_temp['orient'][0])
        #label = 0
        try:
            label = mat_temp['label_split'][0].astype(int) - 1
        except:
            label = -np.Inf
            print('Label information not there')

        # LOAD, ROTATE and CROP the image
        image = numpy.flipud(mat_temp['image'])*127
        # Rotate
        if self.orientation == 'ego':
            image_center = ((image.shape[0]) / 2 - 1, (image.shape[1]) / 2 - 1)
            rot_mat = cv2.getRotationMatrix2D(image_center[1::-1], float(-numpy.rad2deg(orientation)), 1.0)
            image = cv2.warpAffine(image.T, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            image = image.T

        # Crop
        image = crop_numpy_meter(image, bbox_meter_in, self.bbox_meter, self.center)

        # Resize
        if image.shape[0] != self.bbox_pixel[0] or image.shape[1] != self.bbox_pixel[1]:
            image = cv2.resize(image.T, (self.bbox_pixel[1], self.bbox_pixel[0]), interpolation=cv2.INTER_NEAREST)
            image = image.T

        resolution_x = image.shape[1] / self.bbox_meter[1]
        resolution_y = image.shape[0] / self.bbox_meter[0]


        image_out = numpy.tile(image, (self.seq_len, 1, 1))
        
        # DYNAMICS============================================================================================
        # EGO--------------------------------------------------------------------------------------------------
        ego_full = mat_temp['ego']
        ego_temp = ego_full.copy()
        
        if self.orientation == 'ego':
            ego_temp[0:2, :] = rot_points(ego_temp[0:2, :], orientation)
            rect_stack = get_rectangle_multiple_positions(ego_temp[0:2, :], self.rect_width, self.rect_length,
                                                            ego_temp[3, :] + orientation)
        else:
            rect_stack = get_rectangle_multiple_positions(ego_temp[0:2, :], self.rect_width, self.rect_length,
                                                            ego_temp[3, :])
        rect_stack[:, :, 0] = ((rect_stack[:, :, 0] - self.min_x_meter) * resolution_x).astype(int)
        rect_stack[:, :, 1] = (image.shape[0] - 1) - ((rect_stack[:, :, 1] - self.min_y_meter) * resolution_y).astype(
            int)
        
        rect_stack = rect_stack.astype(numpy.int32)
        for idx_in, idx_point in enumerate(ego_temp[7, :].astype(int)):
            if idx_point - 1 >= self.seq_first and idx_point - 1 <= self.seq_last:
                image_out[idx_point - 1 - self.seq_first, :, :] = cv2.fillConvexPoly(
                    image_out[idx_point - 1 - self.seq_first, :, :], rect_stack[idx_in, :, :], (255, 255, 255))

        rect_length_pixel_x = self.rect_length * resolution_x
        rect_length_pixel_y = self.rect_length * resolution_y
        pixel_min_x = (0 - rect_length_pixel_x) / resolution_x + self.min_x_meter
        pixel_max_x = (image.shape[1] + rect_length_pixel_x) / resolution_x + self.min_x_meter
        pixel_min_y = (0 - rect_length_pixel_y) / resolution_y + self.min_y_meter
        pixel_max_y = (image.shape[0] + rect_length_pixel_y) / resolution_y + self.min_y_meter
        # OBJ--------------------------------------------------------------------------------------------------
        obj_temp = mat_temp['obj_list']
        if obj_temp.size != 0:
            for obj in obj_temp[0]:
                time_idx_temp = obj[7, :].astype(int) - 1
                obj_inner = obj[:, (time_idx_temp >= self.seq_first) & (time_idx_temp <= self.seq_last)]
                if  "fieldofview" in self.augmentation_type and random.random() < self.augmentation_type["fieldofview"]:
                    indicator = in_field_of_view(obj_inner,ego_full,self.augmentation_meta['range'],self.augmentation_meta['angle_range'])
                    obj_inner = obj_inner[:,indicator]
                elif "range" in self.augmentation_type:
                    indicator = in_range(obj_inner,ego_full,self.augmentation_meta['range'])
                    obj_inner = obj_inner[:,indicator]
                if obj_inner.size==0:
                    continue

                if self.orientation == 'ego':
                    obj_inner[0:2, :] = rot_points(obj_inner[0:2, :], orientation)
                    obj_inner = obj_inner[:, (obj_inner[0, :] >= pixel_min_x) & (obj_inner[0, :] <= pixel_max_x) & (
                            obj_inner[1, :] >= pixel_min_y) & (obj_inner[1, :] <= pixel_max_y)]

                    rect_stack = numpy.zeros((obj_inner.shape[1], 4, 2))
                    rect_stack[obj_inner[8, :] == 0.0, :, :] = get_rectangle_multiple_positions(
                        obj_inner[0:2, obj_inner[8, :] == 0.0], self.rect_width, self.rect_length,
                        obj_inner[3, obj_inner[8, :] == 0.0] + orientation)
                    rect_stack[obj_inner[8, :] == 1.0, :, :] = get_rectangle_multiple_positions(
                        obj_inner[0:2, obj_inner[8, :] == 1.0], self.square_size, self.square_size,
                        obj_inner[3, obj_inner[8, :] == 1.0] + orientation)
                else:
                    obj_inner = obj_inner[:, (obj_inner[0, :] >= pixel_min_x) & (obj_inner[0, :] <= pixel_max_x) & (
                            obj_inner[1, :] >= pixel_min_y) & (obj_inner[1, :] <= pixel_max_y)]

                    rect_stack = numpy.zeros((obj_inner.shape[1], 4, 2))
                    rect_stack[obj_inner[8, :] == 0.0, :, :] = get_rectangle_multiple_positions(
                        obj_inner[0:2, obj_inner[8, :] == 0.0], self.rect_width, self.rect_length,
                        obj_inner[3, obj_inner[8, :] == 0.0])
                    rect_stack[obj_inner[8, :] == 1.0, :, :] = get_rectangle_multiple_positions(
                        obj_inner[0:2, obj_inner[8, :] == 1.0], self.square_size, self.square_size,
                        obj_inner[3, obj_inner[8, :] == 1.0])

                rect_stack[:, :, 0] = ((rect_stack[:, :, 0] - self.min_x_meter) * resolution_x).astype(int)
                rect_stack[:, :, 1] = image.shape[0] - 1 - (
                            (rect_stack[:, :, 1] - self.min_y_meter) * resolution_y).astype(
                    int)

                rect_stack = rect_stack.astype(numpy.int32)
                idx_vector = (obj_inner[7, :].astype(int) - 1 - self.seq_first).astype(int)
                for idx_in, idx_point in enumerate(idx_vector):
                    image_out[idx_point, :, :] = cv2.fillConvexPoly(image_out[idx_point, :, :],
                                                                    rect_stack[idx_in, :, :], (255, 255, 255))

        image_out = numpy.expand_dims(image_out, axis=0)
        image_out = torch.from_numpy(image_out)
        dictionary_temp = {
            'image': image_out.to(torch.float).div_(2.0),
            'label': torch.tensor(label).long()
        }

        return dictionary_temp
    
    def __getitem__(self, index):
        return self._load_sample_image(self._get_file(index))

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
