from scipy.io import loadmat
import torch
import numpy


from src.utils.rot_points import rot_points
from src.utils.crop_numpy_meter import crop_numpy_meter
from src.utils.get_rectangle import get_rectangle_multiple_positions
from src.utils.in_field_of_view import in_field_of_view
from src.utils.in_range import in_range


def _load_sample_graph(self, filename):   
    mat_temp = loadmat(filename,
                        variable_names=["orient", "ego", "obj_list"],
                        verify_compressed_data_integrity=False)
    orientation = float(-mat_temp['orient'][0])
    frqz = 10
    obs_len = 2
    pred_len = 3
    n_nbrs = 8
    DIM = 4
    
    
    duration = 500 
    end_time_idx = 500
    # MAP============================================================================================
    # Find and append ways in desired BBOX-----------------------------------------------
    # See _load_sample_vector if desired

    # DYNAMICS============================================================================================
    # EGO--------------------------------------------------------------------------------------------------
    ego_temp = mat_temp['ego']
    time_idx_temp = ego_temp[7, :].astype(int) - 1
    ego_hist = ego_temp[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
    ego_pred = ego_temp[:, (time_idx_temp >= self.pred_seq_first) & (time_idx_temp <= self.pred_seq_last)]

    if self.orientation == 'ego':
        ego_hist[0:2, :] = rot_points(ego_hist[0:2, :], orientation)
        ego_pred[0:2, :] = rot_points(ego_pred[0:2, :], orientation)

    ego_xy = ego_hist[0:2, :]
    ego_psi = ego_hist[3, :]
    ego_v = ego_hist[4, :]
    ego_vx = numpy.cos(ego_psi)*ego_v
    ego_vy = numpy.sin(ego_psi)*ego_v
    agent = numpy.stack((ego_xy[0,:], ego_xy[1,:], ego_vx, ego_vy))
    XY_pred = ego_pred[0:2, :]
    # OBJ--------------------------------------------------------------------------------------------------
    def k_nearest_neighbours(agent, obj_temp, k=8, idx=0):
        diff_x = (obj_temp[:, 0, :]**2 - agent[0,:]**2)
        diff_y = (obj_temp[:, 1, :]**2 - agent[1,:]**2)
        diff_abs = numpy.sum((diff_x + diff_y), axis=1)
        idx = numpy.argpartition(diff_abs, k)
        obj_k_nearest = obj_temp[idx[:k], :, :]
        return obj_k_nearest
    # Get objects
    obj_temp = mat_temp['obj_list']
    obj_out = []
    masks = []
    i = 0
    if obj_temp.size != 0:
        # Determine which objects are valid in the required time span
        for obj in obj_temp[0]:
            time_idx_temp = obj[7, :].astype(int) - 1
            obj_inner_hist = obj[:, (time_idx_temp >= self.hist_seq_first) & (time_idx_temp <= self.hist_seq_last)]
            if self.orientation == 'ego':
                obj_inner_hist[0:2, :] = rot_points(obj_inner_hist[0:2, :], orientation)
            if obj_inner_hist.shape[1] == (self.hist_seq_last-self.hist_seq_first+1):
                obj_out.append(obj_inner_hist)           
    # Determine k-nearest neigbours
    if len(obj_out) > n_nbrs:
        nbrs = k_nearest_neighbours(agent, numpy.stack((obj_out[:])), k=8, idx=0)
        nbrs_vx = numpy.cos(nbrs[:, 3])*nbrs[:, 4]
        nbrs_vy = numpy.sin(nbrs[:, 3])*nbrs[:, 4]
    elif len(obj_out) < n_nbrs:
        n_miss = abs(len(obj_out)-n_nbrs)
        nbrs = numpy.stack((obj_out[:]))
        ego_rep = numpy.tile(agent, (n_miss, 1 ,1))
        nbrs_vx = numpy.cos(nbrs[:, 3, :])*nbrs[:, 4, :]
        nbrs_vy = numpy.sin(nbrs[:, 3, :])*nbrs[:, 4, :]
        nbrs = numpy.append(nbrs[:, :2, :], ego_rep[:, :2, :], axis=0)
        nbrs_vx = numpy.append(nbrs_vx, ego_rep[:, 2, :], axis=0)
        nbrs_vy = numpy.append(nbrs_vy, ego_rep[:, 3, :], axis=0)
    else:
        nbrs = numpy.stack((obj_out[:]))
        nbrs_vx = numpy.cos(nbrs[:, 3, :])*nbrs[:, 4, :]
        nbrs_vy = numpy.sin(nbrs[:, 3, :])*nbrs[:, 4, :]
        
    F_X = numpy.concatenate((agent[0, :].reshape(1,-1), nbrs[:, 0, :]), axis=0)
    F_Y = numpy.concatenate((agent[1, :].reshape(1,-1), nbrs[:, 1, :]), axis=0)
    F_VX = numpy.concatenate((agent[2, :].reshape(1,-1), nbrs_vx), axis=0)
    F_VY = numpy.concatenate((agent[3, :].reshape(1,-1), nbrs_vy), axis=0)
    
    return {'F_X': torch.from_numpy(F_X), 'F_Y': torch.from_numpy(F_Y), 
            'F_VX': torch.from_numpy(F_VX), 'F_VY': torch.from_numpy(F_VY), 
            'XY_pred': torch.from_numpy(XY_pred)}