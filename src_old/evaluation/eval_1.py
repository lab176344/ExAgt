from src.evaluation.eval import eval
from sklearn.utils.linear_assignment_ import linear_assignment
import torch

class eval_1(eval):
    def __init__(self, 
                 idx=1,
                 name='Top k clustering accuracy',
                 input_='y_true,y_pred',
                 output='acc.',
                 description='Can calculate top K given the ground truth labels'):
        super().__init__(idx,
                    name,
                    input_,
                    output,
                    description)
    def param_calc(self,output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res