from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
from glob import glob
from pathlib import Path
import torch
import numpy
import cv2
import os

from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions

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
representation_types = ["vector","image_vector","image","image_bound","graph"]

def in_field_of_view(obj, ego, range, angle_range):
    # Align the time
    obj_pos = numpy.zeros((2,obj.shape[1]))
    ego_pos = numpy.zeros((2,obj.shape[1]))
    ego_psi = numpy.zeros((1,obj.shape[1]))
    for idx,time_idx in enumerate(obj[7, :].astype(int)):
        obj_pos[:,idx] = obj[0:2, idx]
        time_match = numpy.where(ego[7,:].astype(int)==time_idx)[0][0]
        ego_pos[:,idx] = ego[0:2,time_match]
        ego_psi[:,idx] = ego[3,time_match]
    
    # Get distance and angle
    distances = numpy.sqrt(numpy.sum((ego_pos-obj_pos)**2,axis=0))
    indicator = numpy.less_equal(distances,range)
    # Check if distance is in range
    # check if angle is in yaw +- angle_range
    angle = numpy.arctan2(obj_pos[1,:]-ego_pos[1,:],obj_pos[0,:]-ego_pos[0,:])
    angle_diff = (ego_psi-angle) % (2*3.14)
    angle_diff[angle_diff>=3.14] -= 2*3.14
    angle_diff = numpy.abs(angle_diff)
    indicator = numpy.bitwise_and(indicator,numpy.less_equal(angle_diff[0],(angle_range/2.0)/180.0*3.14))

    return indicator

def in_range(obj, ego, range):
    # Align the time
    obj_pos = numpy.zeros((2,obj.shape[1]))
    ego_pos = numpy.zeros((2,obj.shape[1]))
    for idx,time_idx in enumerate(obj[7, :].astype(int)):
        obj_pos[:,idx] = obj[0:2, idx]
        time_match = numpy.where(ego[7,:].astype(int)==time_idx)[0][0]
        ego_pos[:,idx] = ego[0:2,time_match]
    
    # Get distance and angle
    distances = numpy.sqrt(numpy.sum((ego_pos-obj_pos)**2,axis=0))
    indicator = numpy.less_equal(distances,range)
    return indicator

class dataset(Dataset):
    def __init__(self, name='argoverse', augmentation_type=[augmentation_types[0]], augmentation_meta=None, representation_type='image', orientation='plain',
                 mode='train', only_edges=False, bbox_meter=[200.0, 200.0], bbox_pixel=[100, 100], center_meter=None,
                 hist_seq_first=0, hist_seq_last=49,pred_seq_first=None,pred_seq_last=None):
        super().__init__()
        self.name = name
        path = (Path(__file__).parent.parent.parent).joinpath('data').joinpath(self.name)
        self.dir = str(path)
        assert representation_type in representation_types, 'representation keyword unknown'
        self.representation_type = representation_type
        self.orientation = orientation
        self.mode = mode
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
        self.rect_width = 2
        self.rect_length = 4.5
        self.square_size = 1.5

        # Check if there is any train/test/val split
        result = [f.name for f in path.rglob("*")]
        if 'train' in result and 'test' in result:
            train_test_split_valid = True
        else:
            print('Please generate train/test split beforehand')
            exit()
        
        self.files = self.augmentation_init()

        self.len = len(self.files)
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

    def _load_sample_vector(self, filename):
        # MAP============================================================================================
        mat_temp = loadmat(filename,
                           variable_names=["center", "bbox", "map_id", "orient", "ego", "obj_list", "way_ids_map"],
                           verify_compressed_data_integrity=False)
        map_id = str(mat_temp['map_id'][0].tolist())
        way_ids_map = mat_temp['way_ids_map'][0]
        center = mat_temp['center']
        orientation = float(-mat_temp['orient'][0])

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

        # DYNAMICS============================================================================================
        # EGO--------------------------------------------------------------------------------------------------
        ego_full = mat_temp['ego']
        ego_temp= ego_full
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
                    obj_out.append(obj_inner)
                    masks.append(mask)

        return {'traj_ego': ego_out, 'traj_objs': obj_out, 'mask_ego': mask_ego, 'mask_objs': masks,
                'map': map_inner, 'mask_map': mask_map}

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
                    obj_out.append(obj_inner)
                    masks.append(mask)

        image_out = numpy.expand_dims(image, axis=0)
        image_out = torch.from_numpy(image_out.copy())
        return {'image': image_out.to(torch.float), 'traj_ego': ego_out, 'traj_objs': obj_out,
                'mask_ego': mask_ego, 'mask_objs': masks}

    def _load_sample_image(self, filename):
        # IMAGE=================================================================================
        mat_temp = loadmat(filename, variable_names=["image", "orient", "image_size_meter", "ego", "obj_list"],
                           verify_compressed_data_integrity=False)
        bbox_meter_in = mat_temp['image_size_meter'].astype(numpy.int16)[0]
        orientation = float(-mat_temp['orient'][0])

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
        # ego / obj[5, :] = a_lon
        # ego / obj[6, :] = a_lat
        ego_velocity = ego_full[4,:].max() - numpy.min(ego_full[4,:][numpy.nonzero(ego_full[4,:])])
        ego_temp = ego_full.copy()
        label = ego_temp[5, :].astype(int)[0] - 1
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
            'label': torch.tensor(label).long(),
            "extra": ego_velocity
        }

        return dictionary_temp

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

    def _load_sample_graph(self, filename):   
        mat_temp = loadmat(filename,
                           variable_names=["orient", "ego", "obj_list"],
                           verify_compressed_data_integrity=False)
        orientation = float(-mat_temp['orient'][0])

        # MAP============================================================================================
        # Find and append ways in desired BBOX-----------------------------------------------
        # See _load_sample_vector if desired

        # DYNAMICS============================================================================================
        # EGO--------------------------------------------------------------------------------------------------
        ego_temp = mat_temp['ego']
        time_idx_temp = ego_temp[7, :].astype(int) - 1
        ego_temp = ego_temp[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
        if self.orientation == 'ego':
            ego_temp[0:2, :] = rot_points(ego_temp[0:2, :], orientation)
        #mask_ego = (ego_temp[0, :] >= self.min_x_meter) & (ego_temp[0, :] <= self.max_x_meter) & (ego_temp[1, :] >= self.min_y_meter) & (ego_temp[1, :] <= self.max_y_meter)
        ego_xy = ego_temp[0:2, :]
        ego_psi = ego_temp[3, :]
        ego_v = ego_temp[4, :]

        # OBJ--------------------------------------------------------------------------------------------------
        # Determine which objects to get (nearest neighbors)

        # Get objects
        obj_temp = mat_temp['obj_list']
        obj_out = []
        masks = []
        if obj_temp.size != 0:
            for obj in obj_temp[0]:
                time_idx_temp = obj[7, :].astype(int) - 1
                obj_inner_hist = obj[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
                obj_inner_pred = obj[:, (time_idx_temp >= self.pred_seq_first) & (time_idx_temp <= self.pred_seq_last)]

                if self.orientation == 'ego':
                    obj_inner_hist[0:2, :] = rot_points(obj_inner_hist[0:2, :], orientation)
                    obj_inner_pred[0:2, :] = rot_points(obj_inner_pred[0:2, :], orientation)

        return {'F_X': F_X, 'F_Y': F_Y, 'F_VX': F_VX, 'F_VY': F_VY, 'XY_pred': XY_pred}

    def __getitem__(self, index):
        if self.representation_type == "image":
            return self._load_sample_image(self.files[index])
        if self.representation_type == "image_vector":
            return self._load_sample_image_vector(self.files[index])
        if self.representation_type == "vector":
            return self._load_sample_vector(self.files[index])
        if self.representation_type == "image_bound":
            return self._load_sample_image_bound(self.files[index])
        if self.representation_type == "graph":
            return self._load_sample_graph(self.files[index])

    def __len__(self):
        return self.len

    def __str__(self):
        return "Loaded {} dataset: {} total samples: {}".format(self.mode,
                                                                self.name,
                                                                len(self))
    def get_num_classes(self):
        if self.name == "openTraffic":
            return 3
        raise RuntimeError("Num classes unknown")

    def get_num_classes(self):
        if self.name == "openTraffic":
            return 3
        if self.name == "highD":
            return 3
        raise RuntimeError

    def _get_description(self):
        # TODO
        description = {'name': self.name}
        return description
    
    def augmentation_init(self):
        if "connectivity" in self.augmentation_type:
            files = sorted(glob(self.dir + "/" + self.mode + "/augmentation/*.mat"))
        else:
            files = sorted(glob(self.dir + "/" + self.mode + "/base/*.mat"))
        return files