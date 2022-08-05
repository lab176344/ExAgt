import torch.nn.functional as F
from src.utils.get_rot_mat import get_rot_mat

def rot_tensor(x, theta):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x