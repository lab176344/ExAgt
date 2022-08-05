from scipy.io import loadmat
import torch
import numpy
import cv2

from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range

def _load_sample_image_vector_merge_recon(self, filename):
    # IMAGE==============================================================================================
    mat_temp = loadmat(filename, variable_names=["image", "orient", "image_size_meter", "ego","label_split"],
                       verify_compressed_data_integrity=False)
    bbox_meter_in = mat_temp['image_size_meter'].astype(numpy.int16)[0]
    orientation = float(-mat_temp['orient'][0])
    label = mat_temp['label_split'][0].astype(int) - 1
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
    ego_out = numpy.zeros((3,self.hist_seq_len))
    ego_lens = ego_temp.shape[1]
    ego_out[:,:ego_lens] = ego_temp[0:3, :]

    image_out = numpy.expand_dims(image, axis=0)
    image_out = torch.from_numpy(image_out.copy())

    # Recon image:
    trajectory = torch.tensor(ego_out)
    resolution = numpy.array(self.bbox_pixel)/numpy.array(self.bbox_meter)
    center = self.center

    trajectory_quant_0 = torch.round((trajectory[0,:]+center[0])*resolution[0]).type(torch.int64) 
    trajectory_quant_1 = torch.round(self.bbox_pixel[1]-(trajectory[1,:]+center[1])*resolution[1]).type(torch.int64) 

    image_merged = torch.zeros([2,image_out.shape[1],image_out.shape[2]])
    image_merged[1,:,:] = image_out[0,:,:]
        
    mask_image = (trajectory_quant_0>(self.bbox_pixel[0]-1)) | (trajectory_quant_1>(self.bbox_pixel[1]-1)) | (trajectory_quant_0<0) | (trajectory_quant_1<0)
        
    image_merged[0,trajectory_quant_1[~mask_image],trajectory_quant_0[~mask_image]] = trajectory[2,~mask_image].float()

    return {'image': image_out.to(torch.float), 'traj_ego': ego_out,'mask_ego': mask_ego, 'ego_lens': ego_lens, 'image_merged': image_merged, 'label':label}