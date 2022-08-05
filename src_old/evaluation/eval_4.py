import logging

import numpy as np
import torch
from sklearn.utils.linear_assignment_ import linear_assignment
from src.evaluation.eval import eval
from src.model.model_3 import BasicBlock, Bottleneck
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class eval_4(eval):
    def __init__(
        self,
        idx=4,
        name='Write embeddings to tensorboard',
        input_='y_true,y_pred',
        output='acc.',
        description='Used for paper experiments; similiar to eval_0 but'
        ' updated to fit into the pipeline'):
        super().__init__(idx, name, input_, output, description)
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, ssl_model, dataloader_train, dataloader_test):
        self.pbar = tqdm(total=1)
        self.pbar.set_description(
            "Training and Evaluating on downstream task".center(50))
        self._evaluate(ssl_model, dataloader_test)
        self.pbar.update()
        self.pbar.close()

    @torch.no_grad()
    def _evaluate(self, ssl_model, dataloader):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """
        representation_dim = 0
        for m in ssl_model.modules():
            if isinstance(m, Bottleneck):
                representation_dim = 2048
                break
            if isinstance(m, BasicBlock):
                representation_dim = 512
                break
        if representation_dim == 0:
            raise NotImplementedError(
                "Representation dim could not be inferred")

        writer = SummaryWriter()

        ssl_model = ssl_model.to(self.device)
        ssl_model.eval()

        z = torch.tensor([]).to(self.device)
        y_true = torch.tensor([]).to(self.device)
        label_img = torch.tensor([]).to(self.device)
        dl = dataloader()
        for batch_idx, (x1, _, y, idx) in tqdm(enumerate(dl)):
            x1 = x1.to(self.device).float()

            y = y.to(self.device)
            _, representations = ssl_model(x1)

            label_img = torch.cat([label_img, x1[:, :,0]])
            z = torch.cat([z, representations])
            y_true = torch.cat([y_true, y])
            if batch_idx * dl.batch_size >= 10000:
                break

        writer.add_embedding(z, metadata=y_true, global_step=1, label_img=label_img)

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
