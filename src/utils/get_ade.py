import torch
from einops import rearrange
def get_ade(X_reshaped, Y_reshaped, gTruthX, gTruthY, maskXnot, maskYnot):
       X_reshaped[maskXnot] = torch.nan
       Y_reshaped[maskYnot] = torch.nan


       pred_traj = torch.cat((X_reshaped,Y_reshaped),dim=3)
       gTruth_traj = torch.cat((gTruthX,gTruthY),dim=3)

       pred_traj = rearrange(pred_traj, 'B O T F -> (B O) T F')
       gTruth_traj = rearrange(gTruth_traj, 'B O T F -> (B O) T F')

       ade = torch.nanmean(torch.sqrt(torch.sum((pred_traj-gTruth_traj)**2,axis=-1)))
       return ade

