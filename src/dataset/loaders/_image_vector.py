from scipy.io import loadmat
import torch
import numpy
import cv2

from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range

def _load_sample_image_vector(self, filename):
    # IMAGE==============================================================================================
    mat_temp = loadmat(filename, variable_names=["image", "orient", "image_size_meter", "ego", "obj_list"],
                        verify_compressed_data_integrity=False)
    bbox_meter_in = mat_temp['image_size_meter'].astype(numpy.int16)[0]
    orientation = float(-mat_temp['orient'][0])

    # LOAD, ROTATE and CROP the image
    image = numpy.flipud(mat_temp['image'])
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

    # DYNAMICS============================================================================================
    # EGO--------------------------------------------------------------------------------------------------
    ego_full = mat_temp['ego']
    ego_temp = ego_full.copy()
    time_idx_temp = ego_temp[7, :].astype(int) - 1
    ego_temp = ego_temp[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
    if self.orientation == 'ego':
        ego_temp[0:2, :] = rot_points(ego_temp[0:2, :], orientation)
    mask_ego = (ego_temp[0, :] >= self.min_x_meter) & (ego_temp[0, :] <= self.max_x_meter) & (
            ego_temp[1, :] >= self.min_y_meter) & (ego_temp[1, :] <= self.max_y_meter)
    ego_out = ego_temp[0:3, :]

    # OBJ--------------------------------------------------------------------------------------------------
    obj_temp = mat_temp['obj_list']
    obj_out = []
    masks = []
    if obj_temp.size != 0:
        for obj in obj_temp[0]:
            time_idx_temp = obj[7, :].astype(int) - 1
            obj_inner = obj[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
            if "fieldofview" in self.augmentation_type:
                indicator = in_field_of_view(obj_inner,ego_full,self.augmentation_meta['range'],self.augmentation_meta['angle_range'])
                obj_inner = obj_inner[:,indicator]
            elif "range" in self.augmentation_type:
                indicator = in_range(obj_inner,ego_full,self.augmentation_meta['range'])
                obj_inner = obj_inner[:,indicator]
            if obj_inner.size==0:
                continue
            if self.orientation == 'ego':
                obj_inner[0:2, :] = rot_points(obj_inner[0:2, :], orientation)
            mask = (obj_inner[0, :] >= self.min_x_meter) & (obj_inner[0, :] <= self.max_x_meter) & (
                    obj_inner[1, :] >= self.min_y_meter) & (obj_inner[1, :] <= self.max_y_meter)
            if numpy.any(mask):
                obj_out.append(torch.from_numpy(obj_inner).t())
                masks.append(mask)
    obj_out.append(torch.from_numpy(ego_temp).t())
    padded_seq = torch.nn.utils.rnn.pad_sequence(obj_out, batch_first=True)
    padded_seq = padded_seq[:-1:,:,:]
    _ = obj_out.pop()
    seq_lens = [obj_temp.shape[0] for obj_temp in obj_out]
    image_out = numpy.expand_dims(image, axis=0)
    image_out = torch.from_numpy(image_out.copy())
    return {'image': image_out.to(torch.float), 'traj_ego': ego_out, 'traj_objs': padded_seq,'traj_objs_lens': seq_lens,
            'mask_ego': mask_ego, 'mask_objs': masks}