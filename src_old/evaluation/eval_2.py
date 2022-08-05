import logging

import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluation.eval import eval
from src.model.model_3 import BasicBlock, Bottleneck
from src.utils.average_meter import AverageMeter
from tqdm import tqdm


class eval_2(eval):
    def __init__(self,
                 idx=2,
                 name='Linear classifier evaluation',
                 input_='y_true,y_pred',
                 output='acc.',
                 description='todo',
                 n_classes=3,
                 epochs=1):
        super().__init__(idx, name, input_, output, description)
        self.n_epochs = epochs
        self.n_classes = n_classes
        self.pbar = None
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, ssl_model, dataloader_train, dataloader_test):
        steps_train = len(dataloader_train.dataset[0]
                          ) / dataloader_train.batch_size * self.n_epochs
        steps_test = len(dataloader_test.dataset[0]) / dataloader_test.batch_size
        self.pbar = tqdm(total=int(steps_train + steps_test))
        self.pbar.set_description(
            "Training and Evaluating on downstream task".center(50))
        self._evaluate(ssl_model, dataloader_train, dataloader_test)
        self.pbar.close()

    def _evaluate(self, ssl_model, dataloader_train, dataloader_test):
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
            raise NotImplementedError("Representation dim could not be inferred")

        transfer_model = LinearClassifier(representation_dim,
                                          self.n_classes).to(self.device)
        ssl_model = ssl_model.to(self.device)
        self._train_transfer_task(ssl_model, transfer_model, dataloader_train)
        self._test_transfer_task(ssl_model, transfer_model, dataloader_test)

    def _train_transfer_task(self, ssl_model, transfer_model, dataloader):
        """Starts training for the downstream task
        """
        loss_fc = nn.CrossEntropyLoss()
        optimizer = optim.Adam(transfer_model.parameters())
        ssl_model.eval()
        for epoch in range(self.n_epochs):
            loss_record = AverageMeter()
            dl = dataloader()
            for batch_idx, (x1, _, y, idx) in enumerate(dl):
                x1 = x1.float()
                with torch.no_grad():
                    _, representation = ssl_model(x1.to(self.device))
                pred = transfer_model(representation)
                loss = loss_fc(pred, y.to(self.device))
                loss_record.update(loss.item(), x1.size(0))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.pbar.update(1)
            logging.info(
                "Downstream task training epoch {}/{} loss: {}".format(
                    epoch + 1, self.n_epochs, loss_record.avg))


    def _test_transfer_task(self, ssl_model, transfer_model, dataloader):
        """Evaluates accuracy on a previously trained linear classifier
        """
        correct = 0
        n_samples = 0
        ssl_model.eval()
        for batch_idx, (x1, _, y, idx) in enumerate(dataloader()):
            with torch.no_grad():
                x1 = x1.float()
                _, representation = ssl_model(x1.to(self.device))
                pred = transfer_model(representation)
                pred = torch.argmax(pred, dim=1)
                correct += pred.eq(y.to(
                    self.device).view_as(pred)).sum().item()
                n_samples += pred.shape[0]
                self.pbar.update(1)

        acc = correct / len(dataloader.dataset[0])
        print("Downstream task accuracy {}".format(acc))
        logging.info("Downstream task test accuracy: {}".format(acc))
        return acc


class LinearClassifier(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.layer = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        return self.layer(x)
