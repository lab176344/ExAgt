from scipy.io import loadmat
import numpy

from src.utils.rot_points import rot_points
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range

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