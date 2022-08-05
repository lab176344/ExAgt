from os import error
import torch

def crop_tensor_3d(tensor,size_pixel_out,center_location):
    input_size = torch.tensor(tensor.shape[1:])
    pixel_min = torch.floor(center_location -size_pixel_out/2.0).to(torch.int)
    pixel_max = torch.floor(center_location +size_pixel_out/2.0).to(torch.int)
    cropped_tensor = torch.tensor([])
    if torch.any(torch.lt(pixel_min,0)) or torch.any(torch.gt(pixel_max,input_size)):
        error('TODO')
    else:
        cropped_tensor = tensor[:,pixel_min[0]:pixel_max[0],pixel_min[1]:pixel_max[1]]
    return cropped_tensor

def crop_tensor_4d(tensor,size_pixel_out,center_location):
    input_size = torch.tensor(tensor.shape[2:])
    pixel_min = torch.floor(center_location -size_pixel_out/2.0).to(torch.int)
    pixel_max = torch.floor(center_location +size_pixel_out/2.0).to(torch.int)
    cropped_tensor = torch.tensor([])
    if torch.any(torch.lt(pixel_min,0)) or torch.any(torch.gt(pixel_max,input_size)):
        error('TODO')
    else:
        cropped_tensor = tensor[:,:,pixel_min[0]:pixel_max[0],pixel_min[1]:pixel_max[1]]
    return cropped_tensor