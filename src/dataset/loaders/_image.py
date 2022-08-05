from scipy.io import loadmat
import torch
import numpy
import cv2
import numpy as np
from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range

def _load_sample_image(self, filename):
    # IMAGE=================================================================================
    mat_temp = loadmat(filename, variable_names=["image", "orient", "image_size_meter", "ego", "obj_list","label_split"],
                        verify_compressed_data_integrity=False)
    bbox_meter_in = mat_temp['image_size_meter'].astype(numpy.int16)[0]
    orientation = float(-mat_temp['orient'][0])
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

    if self.only_edges:
        image = cv2.Canny(image, 0, 1)
        fillColor = (255.0, 255.0, 255.0)
    else:
        fillColor = (0.0, 0.0, 0.0)
    image_out = numpy.tile(image, (self.hist_seq_len, 1, 1))
    
    # DYNAMICS============================================================================================
    # EGO--------------------------------------------------------------------------------------------------
    ego_full = mat_temp['ego']
    ego_temp = ego_full.copy()
    
    # TODO generic label mapping
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
        if idx_point - 1 >= self.hist_seq_first and idx_point - 1 <= self.hist_seq_last:
            image_out[idx_point - 1 - self.hist_seq_first, :, :] = cv2.fillConvexPoly(
                image_out[idx_point - 1 - self.hist_seq_first, :, :], rect_stack[idx_in, :, :], (255, 255, 255))

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
            obj_inner = obj[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
            if  "fieldofview" in self.augmentation_type:
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
            idx_vector = (obj_inner[7, :].astype(int) - 1 - self.hist_seq_first).astype(int)
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