from scipy.io import loadmat
import torch
import numpy
import cv2

from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range

def _rotate_image(rot_mat,image):
    image = cv2.warpAffine(image.T, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return image.T

def _resize_image(image,bbox_pixel):
    if image.shape[0] != bbox_pixel[0] or image.shape[1] != bbox_pixel[1]:
        image = cv2.resize(image.T, (bbox_pixel[1], bbox_pixel[0]), interpolation=cv2.INTER_NEAREST)
        image = image.T
    return image

def _load_sample_image_multichannel_vector(self, filename):
    # IMAGE==============================================================================================
    #mat_temp = loadmat(filename, variable_names=["image", "imCenter", "imInterSection", "imEgoRoute", "imhasTrafficControl", 
    #                                             "orient", "image_size_meter", "ego", "obj_list"],
    #                    verify_compressed_data_integrity=False)
    mat_temp = loadmat(filename, variable_names=["image","orient", "image_size_meter", "ego", "obj_list"],
                        verify_compressed_data_integrity=False)
    print(mat_temp.keys())
    bbox_meter_in = mat_temp['image_size_meter'].astype(numpy.int16)[0]
    orientation = float(-mat_temp['orient'][0])

    # LOAD, ROTATE and CROP the image
    image = numpy.flipud(mat_temp['image'])
    # image_center_line = numpy.flipud(mat_temp['imCenter'])
    # image_intersection = numpy.flipud(mat_temp['imInterSection'])
    # image_ego_route = numpy.flipud(mat_temp['imEgoRoute'])
    # image_traffic_element = numpy.flipud(mat_temp['imhasTrafficControl'])
    
    # Rotate
    if self.orientation == 'ego':
        image_center = ((image.shape[0]) / 2 - 1, (image.shape[1]) / 2 - 1)
        rot_mat = cv2.getRotationMatrix2D(image_center[1::-1], float(-numpy.rad2deg(orientation)), 1.0)
        #image = cv2.warpAffine(image.T, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        #image = image.T
        image = _rotate_image(rot_mat,image)
        # image_center_line = _rotate_image(rot_mat,image_center_line)
        # image_intersection = _rotate_image(rot_mat,image_intersection)
        # image_ego_route = _rotate_image(rot_mat,image_ego_route)
        # image_traffic_element = _rotate_image(rot_mat,image_traffic_element)

    # Crop
    image = crop_numpy_meter(image, bbox_meter_in, self.bbox_meter, self.center)
    # image_center_line = crop_numpy_meter(image_center_line, bbox_meter_in, self.bbox_meter, self.center)
    # image_intersection = crop_numpy_meter(image_intersection, bbox_meter_in, self.bbox_meter, self.center)
    # image_ego_route = crop_numpy_meter(image_ego_route, bbox_meter_in, self.bbox_meter, self.center)
    # image_traffic_element = crop_numpy_meter(image_traffic_element, bbox_meter_in, self.bbox_meter, self.center)
    
    
    
    # Resize
    image = _resize_image(image, self.bbox_pixel)
    # image_center_line = _resize_image(image_center_line, self.bbox_pixel)
    # image_intersection = _resize_image(image_intersection, self.bbox_pixel)
    # image_ego_route = _resize_image(image_ego_route, self.bbox_pixel)
    # image_traffic_element = _resize_image(image_traffic_element, self.bbox_pixel)

    # DYNAMICS============================================================================================
    # EGO--------------------------------------------------------------------------------------------------
    ego_full = mat_temp['ego']
    time_idx_temp = ego_full[7, :].astype(int) - 1
    ego_temp = ego_full[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
    if self.orientation == 'ego':
        ego_temp[0:2, :] = rot_points(ego_temp[0:2, :], orientation)
    ego_out_hist = torch.from_numpy(ego_temp[0:3, :]).t()
    mask_ego_hist = (ego_temp[0, :] >= self.min_x_meter) & (ego_temp[0, :] <= self.max_x_meter) & (
            ego_temp[1, :] >= self.min_y_meter) & (ego_temp[1, :] <= self.max_y_meter)
    
    
    ego_temp = ego_full[:, (time_idx_temp >= self.pred_seq_first) & (time_idx_temp <= self.pred_seq_last)]
    if self.orientation == 'ego':
        ego_temp[0:2, :] = rot_points(ego_temp[0:2, :], orientation)
    ego_out_pred = torch.from_numpy(ego_temp[0:3, :]).t()
    mask_ego_pred = (ego_temp[0, :] >= self.min_x_meter) & (ego_temp[0, :] <= self.max_x_meter) & (
            ego_temp[1, :] >= self.min_y_meter) & (ego_temp[1, :] <= self.max_y_meter)
    
    

    # OBJ--------------------------------------------------------------------------------------------------
    obj_temp = mat_temp['obj_list']
    obj_out_hist = []
    obj_out_pred = []
    masks_obj_hist = []
    masks_obj_pred = []

    if obj_temp.size != 0:
        for obj in obj_temp[0]:
            time_idx_temp = obj[7, :].astype(int) - 1
            # History
            obj_inner = obj[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
            if obj_inner.size == 0:
                continue
            if "fieldofview" in self.augmentation_type:
                indicator = in_field_of_view(obj_inner,ego_full,self.augmentation_meta['range'],self.augmentation_meta['angle_range'])
                obj_inner = obj_inner[:,indicator]
            elif "range" in self.augmentation_type:
                indicator = in_range(obj_inner,ego_full,self.augmentation_meta['range'])
                obj_inner = obj_inner[:,indicator]
            

            
            # Prediction
            obj_inner_pred = obj[:, (time_idx_temp >= self.pred_seq_first) & (time_idx_temp <= self.pred_seq_last)]
            if "fieldofview" in self.augmentation_type:
                indicator = in_field_of_view(obj_inner_pred,ego_full,self.augmentation_meta['range'],self.augmentation_meta['angle_range'])
                obj_inner_pred = obj_inner_pred[:,indicator]
            elif "range" in self.augmentation_type:
                indicator = in_range(obj_inner,ego_full,self.augmentation_meta['range'])
                obj_inner_pred = obj_inner_pred[:,indicator]
            
            if obj_inner_pred.size==0 or obj_inner.size==0:
                continue
            
            # History            
            if self.orientation == 'ego':
                obj_inner[0:2, :] = rot_points(obj_inner[0:2, :], orientation)
                
            mask = (obj_inner[0, :] >= self.min_x_meter) & (obj_inner[0, :] <= self.max_x_meter) & (
                    obj_inner[1, :] >= self.min_y_meter) & (obj_inner[1, :] <= self.max_y_meter)
            if numpy.any(mask):
                masks_obj_hist.append(mask)
                obj_out_hist.append(torch.from_numpy(obj_inner[0:3,:]).t())

            # Prediction
            if self.orientation == 'ego':
                obj_inner_pred[0:2, :] = rot_points(obj_inner_pred[0:2, :], orientation)
            if numpy.any(mask):
                masks_obj_pred.append(mask)
                obj_out_pred.append(torch.from_numpy(obj_inner_pred[0:3,:]).t())
    
    # TODO introduce may seq length 
    obj_out_hist.append(ego_out_hist) # obj_out_hist = history seq. length x features (x,y,timestamp) unpadded
    obj_out_pred.append(ego_out_pred) # obj out pred = history seq. length x features (x,y,timestamp) unpadded
    obj_out_hist_seq_lens = [obj_temp.shape[0] for obj_temp in obj_out_hist] # history seq. length x 1 as a list
     
    # Padding done here to make sure that the sequences are of the same length
    obj_out_hist = torch.nn.utils.rnn.pad_sequence(obj_out_hist, batch_first=True) # num_objects x max_seq_len x features (x,y,timestamp) # padded
    
    #obj_out_hist_seq_lens = torch.from_numpy(numpy.array(obj_out_hist_seq_lens))
    
    obj_decoder_in = [obj_temp[obj_out_hist_seq_lens[ids]-1,:] for ids, obj_temp in enumerate(obj_out_hist)]
    obj_decoder_in = torch.nn.utils.rnn.pad_sequence(obj_decoder_in, batch_first=True) # num_objects x  features (x,y,timestamp) # padded, takes the last element from history
    obj_out_pred_seq_lens = [obj_temp.shape[0] for obj_temp in obj_out_pred] # pred seq. length x 1 as a list

    obj_out_pred = torch.nn.utils.rnn.pad_sequence(obj_out_pred, batch_first=True,padding_value=torch.nan) # num_objects x max_seq_len x features (x,y,timestamp) # padded,with nan to filter during loss

    #obj_out_pred_seq_lens = torch.from_numpy(numpy.array(obj_out_pred_seq_lens))
    assert obj_out_hist.shape[0] == obj_out_pred.shape[0]
    
    image_out = numpy.expand_dims(image, axis=0) # 1  x H x W
    # image_out_center_line = numpy.expand_dims(image_center_line, axis=0)
    # image_out_intersection = numpy.expand_dims(image_intersection, axis=0)
    # image_out_ego_route = numpy.expand_dims(image_ego_route, axis=0)
    # image_out_had_traffic_control = numpy.expand_dims(image_traffic_element, axis=0)
    
    image_out = torch.from_numpy(image_out.copy())
    # image_out_center_line = torch.from_numpy(image_out_center_line.copy())
    # image_out_intersection = torch.from_numpy(image_out_intersection.copy())
    # image_out_ego_route = torch.from_numpy(image_out_ego_route.copy())
    # image_out_had_traffic_control = torch.from_numpy(image_out_had_traffic_control.copy())
    
    #image_out_multichannel = torch.cat((image_out,image_out_center_line,image_out_intersection,image_out_ego_route,image_out_had_traffic_control),dim=0)
    # image_out_multichannel: 0: fullmap x 1: center_line x 2: intersection x 3: ego_route x 4: traffic_control
    return {'image': image_out.to(torch.float), 'traj_hist_obj': obj_out_hist.to(torch.float), 'traj_pred_obj': obj_out_pred.to(torch.float),
            'traj_hist_obj_seq_lens': obj_out_hist_seq_lens, 'traj_pred_obj_seq_lens': obj_out_pred_seq_lens, 'obj_decoder_in': obj_decoder_in.to(torch.float)}