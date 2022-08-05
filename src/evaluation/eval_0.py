from src.evaluation.eval import eval
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import torch

class eval_0(eval):
    def __init__(self, 
                 idx=0,
                 name='Unsupervised clustering accuracy',
                 input_='y_true,y_pred',
                 output='acc.',
                 description='Unsupervised clustering acc. can be used to evaluate clustering assignments given the ground truth labels'):
        super().__init__(idx,
                    name,
                    input_,
                    output,
                    description)
    def param_calc(self,y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed

        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`

        # Return
            accuracy, in [0,1]
        """
        with torch.no_grad():

            y_true = y_true.astype(np.int64)
            assert y_pred.size == y_true.size
            D = max(y_pred.max(), y_true.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(y_pred.size):
                w[y_pred[i], y_true[i]] += 1
            ind = linear_assignment(w.max() - w)
            return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size