import torch
from src.utils.crop_tensor import crop_tensor_3d,crop_tensor_4d
def crop_tensor_meter_3d(tensor, size_meter_in, size_meter_out, center_location_out):
    resolution = torch.true_divide(size_meter_in,torch.tensor(tensor.shape[1:]))
    size_pixel_out = torch.round(size_meter_out/resolution)
    if center_location_out is None:
        center_location = torch.round(torch.tensor(tensor.shape[1:])/2.0)
    else:
        center_location = torch.round(torch.tensor(tensor.shape[1:])/2.0 + size_pixel_out/2.0 - center_location_out/resolution)
    cropped_tensor = crop_tensor_3d(tensor,size_pixel_out,center_location)
    return cropped_tensor

def crop_tensor_meter_4d(tensor, size_meter_in, size_meter_out, center_location_out):
    resolution = torch.true_divide(size_meter_in,torch.tensor(tensor.shape[2:]))
    size_pixel_out = torch.round(size_meter_out/resolution)
    if center_location_out is None:
        center_location = torch.round(torch.tensor(tensor.shape[2:])/2.0)
    else:
        center_location = torch.round(torch.tensor(tensor.shape[2:])/2.0 + size_pixel_out/2.0 - center_location_out/resolution)
    cropped_tensor = crop_tensor_4d(tensor,size_pixel_out,center_location)
    return cropped_tensor