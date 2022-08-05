import torch.nn.functional as F
def resize_tensor_3d(tensor, size, algorithm='nearest'):
    # Check input dim
    if tensor.ndim == 3:
        # Single image
        resized_tensor = F.interpolate(tensor.view(1,tensor.shape[0],tensor.shape[1],tensor.shape[2]), size=(int(size.data[0]),int(size.data[1])),mode=algorithm)
        return resized_tensor.view(tensor.shape[0],size.data[0],size.data[1])
    elif tensor.ndim ==4:
        # batch of image
        resized_tensor = F.interpolate(tensor, size=(int(size.data[0]),int(size.data[1])),mode=algorithm)
        return resized_tensor

def resize_tensor_4d(tensor, size, algorithm='nearest'):
    # Check input dim
    if tensor.ndim == 4:
        # Single image
        resized_tensor = F.interpolate(tensor.view(1,tensor.shape[0],tensor.shape[1],tensor.shape[2],tensor.shape[3]), size=(int(tensor.shape[1]),int(size.data[0]),int(size.data[1])),mode=algorithm)
        return resized_tensor.view(tensor.shape[0],tensor.shape[1],size.data[0],size.data[1])
    elif tensor.ndim ==5:
        # batch of image
        resized_tensor = F.interpolate(tensor, size=(int(size.data[0]),int(size.data[1])),mode=algorithm)
        return resized_tensor