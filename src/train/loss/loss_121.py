import torch
from src.train.loss.loss import loss

class loss_121(loss):
    def __init__(self,
                idx = 121,
                name = 'Delta loss',
                description = 'Used to control the rate of the outputs, eg. jerk, acceleration, steering rate, etc',
                input_ = 'dynamic outputs from the model and thresholds',
                output = 'loss',
                thres_dynamic_1=None,
                thres_dynamic_2=None
                ) -> None:
        super().__init__(idx,name,description,input_,output)
        self.thres_dynamic_1 = thres_dynamic_1
        self.thres_dynamic_2 = thres_dynamic_2
    def forward(self, dynamic_1 = None, dynamic_2 = None):
        

        delta_dynamic_1 = torch.diff(dynamic_1)/0.1 # 0.1 is the sample rate change this to a param later
        delta_dynamic_2 = torch.diff(dynamic_2)/0.1 # 0.1 is the sample rate change this to a param later
        delta_dynamic_1 = torch.abs(delta_dynamic_1)
        delta_dynamic_2 = torch.abs(delta_dynamic_2)
        max_delta_1 = torch.max(delta_dynamic_1,axis=1)
        max_delta_2 = torch.max(delta_dynamic_2,axis=1)
        thre_boolean_dynamic_1 = delta_dynamic_1 > self.thres_dynamic_1
        thre_boolean_dynamic_2 = delta_dynamic_2 > self.thres_dynamic_2

        loss_dynamic_1 = torch.sum(delta_dynamic_1[thre_boolean_dynamic_1])
        loss_dynamic_2 = torch.sum(delta_dynamic_2[thre_boolean_dynamic_2])
        #thre_boolean_dynamic_1 = thre_boolean_dynamic_1.float()
        #thre_boolean_dynamic_2 = thre_boolean_dynamic_2.float()

        #loss_dynamic_1 = torch.mean(torch.sum(thre_boolean_dynamic_1,axis=1))
        #loss_dynamic_2 = torch.mean(torch.sum(thre_boolean_dynamic_2,axis=1))

        loss_dynamic = loss_dynamic_1 + loss_dynamic_2
        return loss_dynamic,max_delta_1[0],max_delta_2[0]