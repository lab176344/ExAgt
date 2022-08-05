from scipy.io import loadmat
import torch
import numpy
import cv2

from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range

def _load_sample_image_bound(self, filename):
    # IMAGE=================================================================================
    mat_temp = loadmat(filename,
                        variable_names=["center", "bbox", "map_id", "way_ids_map", "orient", "image_size_meter",
                                        "ego", "obj_list"], verify_compressed_data_integrity=False)
    map_id = str(mat_temp['map_id'][0].tolist())
    way_ids_map = mat_temp['way_ids_map'][0]
    center = mat_temp['center']
    bbox_meter_in = mat_temp['image_size_meter'].astype(numpy.int16)[0]
    orientation = float(-mat_temp['orient'][0])
    resolution = numpy.array(self.bbox_pixel) / numpy.array(self.bbox_meter)

    bbox_origin = [center[0] - self.bbox_meter[0] / 2.0, center[0] + self.bbox_meter[0] / 2.0,
                    center[1] - self.bbox_meter[1] / 2.0, center[1] + self.bbox_meter[1] / 2.0]
    way_ids, way_inside, way_outside, ways_bbox = self.map[map_id]

    # Find and append ways in desired BBOX-----------------------------------------------
    b_2_left_g_b_1_right = (ways_bbox[:, 0] > bbox_origin[1])
    b_2_right_l_b_1_left = (ways_bbox[:, 1] < bbox_origin[0])
    b_2_top_g_l_1_bottom = (ways_bbox[:, 3] < bbox_origin[2])
    b_2_bottom_g_b_1_top = (ways_bbox[:, 2] > bbox_origin[3])

    include = ~(b_2_left_g_b_1_right | b_2_right_l_b_1_left | b_2_top_g_l_1_bottom | b_2_bottom_g_b_1_top)

    way_inside = way_inside[include]
    way_outside = way_outside[include]
    way_ids = way_ids[include]

    map_inner = []
    mask_map = []
    for way_id, inside, outside in zip(way_ids, way_inside, way_outside):
        if way_id in way_ids_map:
            inside = inside[:, :2].T - center
            outside = outside[:, :2].T - center
            if self.orientation == 'ego':
                inside = rot_points(inside, orientation)
                outside = rot_points(outside, orientation)

            mask_inside = (inside[0, :] >= self.min_x_meter) & (inside[0, :] <= self.max_x_meter) & (
                        inside[1, :] >= self.min_y_meter) & (inside[1, :] <= self.max_y_meter)
            mask_outside = (outside[0, :] >= self.min_x_meter) & (outside[0, :] <= self.max_x_meter) & (
                        outside[1, :] >= self.min_y_meter) & (outside[1, :] <= self.max_y_meter)
            if numpy.any(mask_inside) or numpy.any(mask_outside):
                map_inner.append(inside)
                mask_map.append(mask_inside)
                map_inner.append(outside)
                mask_map.append(mask_outside)

    # Create 1xwxh zero numpy array
    # transform cooridnates from meters to pixels (map_inner)
    # use cv2.lineprint (image, line)
    # LOAD, ROTATE and CROP the image

    image = numpy.zeros((self.bbox_pixel[0], self.bbox_pixel[1]), dtype=numpy.uint8)
    for line in map_inner:
        line1 = (line[0, :] + self.center[0]) * resolution[0]
        line2 = self.bbox_pixel[1] - (line[1, :] + self.center[1]) * resolution[1]
        pointArray = [[x, y] for x, y in zip(line1, line2)]
        pointArray = numpy.int32(pointArray)
        image = cv2.polylines(image, [pointArray], 1, (255.0, 255.0, 255.0))

    # image = numpy.flipud(mat_temp['image'])
    # # Rotate
    # if self.orientation=='ego':
    #     image_center= ((image.shape[0]) / 2 -1,(image.shape[1]) / 2-1)
    #     rot_mat     = cv2.getRotationMatrix2D(image_center[1::-1], float(-numpy.rad2deg(orientation)), 1.0)
    #     image       = cv2.warpAffine(image.T, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    #     image       = image.T

    # # Crop
    # image = crop_numpy_meter(image, bbox_meter_in, self.bbox_meter,self.center)

    # # Resize
    # if image.shape[0] != self.bbox_pixel[0] or image.shape[1] != self.bbox_pixel[1]:
    #     image = cv2.resize(image.T,(self.bbox_pixel[1],self.bbox_pixel[0]),interpolation=cv2.INTER_NEAREST)
    #     image = image.T

    resolution_x = image.shape[1] / self.bbox_meter[1]
    resolution_y = image.shape[0] / self.bbox_meter[0]

    # image   *= 2
    # image   += 1
    image_out = numpy.tile(image, (self.hist_seq_len, 1, 1))

    # DYNAMICS============================================================================================
    # EGO--------------------------------------------------------------------------------------------------
    ego_temp = mat_temp['ego']
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
            if "fieldofview" in self.augmentation_type:
                indicator = in_field_of_view(obj_inner,ego_full,self.augmentation_meta['range'],self.augmentation_meta['angle_range'])
                obj_inner = obj_inner[:,indicator]
            elif "range" in self.augmentation_type:
                indicator = in_range(obj_inner,ego_full)
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
                        (rect_stack[:, :, 1] - self.min_y_meter) * resolution_y).astype(int)

            rect_stack = rect_stack.astype(numpy.int32)
            idx_vector = (obj_inner[7, :].astype(int) - 1 - self.hist_seq_first).astype(int)
            for idx_in, idx_point in enumerate(idx_vector):
                image_out[idx_point, :, :] = cv2.fillConvexPoly(image_out[idx_point, :, :],
                                                                rect_stack[idx_in, :, :], (255, 255, 255))

    image_out = numpy.expand_dims(image_out, axis=0)
    image_out = torch.from_numpy(image_out)
    return {'image': image_out.to(torch.float).div_(2.0)}