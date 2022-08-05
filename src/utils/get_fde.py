
import torch
def get_fde(X_reshaped, Y_reshaped, gTruthX, gTruthY, x_traj_pred_len, maskXnot, maskYnot):
      pred_fde_incides = torch.cumsum(torch.Tensor(x_traj_pred_len),0).long()
      pred_fde_incides = pred_fde_incides.to(X_reshaped.device)
      x_fde_pred = X_reshaped[~maskXnot]
      x_fde_pred = torch.take(x_fde_pred, pred_fde_incides-1)
      y_fde_pred = Y_reshaped[~maskYnot]
      y_fde_pred = torch.take(y_fde_pred, pred_fde_incides-1)
      x_fde_true = gTruthX[~maskXnot]
      x_fde_true = torch.take(x_fde_true, pred_fde_incides-1)
      y_fde_true = gTruthY[~maskYnot]
      y_fde_true = torch.take(y_fde_true, pred_fde_incides-1)
      mean_FDE = torch.sum(torch.sqrt((x_fde_pred-x_fde_true)**2 + (y_fde_pred-y_fde_true)**2))
      return mean_FDE/x_fde_pred.shape[0]