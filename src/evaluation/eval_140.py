import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.evaluation.eval import eval
from src.utils.average_meter import AverageMeter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
class eval_140(eval):
    def __init__(self, 
                 idx=140,
                 name='MSE prediction accuracy',
                 input_='y_true,y_pred',
                 output='acc.',
                 description='todo',
                 pred_ = 50,
                  ):
        super().__init__(idx,
                    name,
                    input_,
                    output,
                    description)
        self.pred_frames = pred_

    def __call__(self, ssl_model, dataloader_train, dataloader_test):
        self._evaluate(ssl_model, dataloader_train, dataloader_test)

    def _evaluate(self, ssl_model, dataloader_train, dataloader_test):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """
        def mse_eval(out_1 = None, out_2 = None):
            PRED_FRAMES = self.pred_frames
            RCL_y = F.mse_loss(out_1[PRED_FRAMES:], out_2[PRED_FRAMES:], reduction = 'sum')
            RCL_x = F.mse_loss(out_1[:PRED_FRAMES], out_2[:PRED_FRAMES], reduction = 'mean')
            RCL = RCL_y + RCL_x
            return RCL
        
        loss_record = AverageMeter()
        for batch_idx, (F_x, F_y, F_vx, F_vy,
                            xy_pred) in enumerate(dataloader_test(epoch=0)):
            v0 = F_vx[0,-1]
            y_hat, z, F_re = ssl_model(F_x.transpose(0,1).to(device).float(), F_y.transpose(0,1).to(device).float(), 
                                   F_vx.transpose(0,1).to(device).float(), F_vy.transpose(0,1).to(device).float(), v0.to(device).float())
            loss = mse_eval(torch.cat((xy_pred[0, 0, :], xy_pred[0, 1, :])).to(device), 
                            y_hat[0].to(device))
            loss_record.update(loss.item(), F_x.size(0))
            logging.info("Downstream task training loss: {}".format(loss_record.avg))
        return loss_record.avg
