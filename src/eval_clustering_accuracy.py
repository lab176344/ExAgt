import logging

import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.utils.linear_assignment_ import linear_assignment
from tqdm import tqdm
from scipy.cluster.hierarchy import fcluster
import fastcluster as fc
import umap
import json

class eval_clustering_accuracy(object):
    def __init__(self):
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, ssl_model, dataloader_test):

        self.pbar = tqdm(total=1)
        self.pbar.set_description(
            "Training and Evaluating on downstream task".center(50))
        self._evaluate(ssl_model, dataloader_test)
        self.pbar.update()
        self.pbar.close()

    def forward_adapter(self, model, x):
        representation = model._forward_backbone(x)
        return representation
       
    @torch.no_grad()
    def _evaluate(self, ssl_model, dataloader):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """
        ssl_model.eval()
        representation_dim = 512
        if representation_dim == 0:
            raise NotImplementedError(
                "Representation dim could not be inferred")

        n_classes = 26  
        ssl_model = ssl_model.to(self.device)

        z = torch.tensor([]).to(self.device)
        y_true = torch.tensor([]).to(self.device)

        for batch_idx, (x1, _, y, idx) in tqdm(enumerate(dataloader())):
            x1 = x1.to(self.device).float()
            y = y.to(self.device)
            representations = self.forward_adapter(ssl_model,x1)
            z = torch.cat([z, representations])
            y_true = torch.cat([y_true, y])


        Tree = fc.linkage(z.cpu().numpy(), method='average', metric='euclidean')

        cluster_assignments = fcluster(Tree, n_classes, criterion='maxclust')

        acc = self.calc_clustering_acc(y_true.cpu().numpy(),
                                       cluster_assignments)
        print("Unsupervised clustering assignment is: {}".format(acc))
        logging.info("Unsupervised clustering assignment is {}".format(acc))

    def calc_clustering_acc(self, y_true, y_pred):
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
