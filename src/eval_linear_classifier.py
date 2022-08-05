import logging

import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.average_meter import AverageMeter
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report


class eval_linear_classifier(object):
    def __init__(self,
                 idx=2,
                 name='Linear classifier evaluation',
                 input_='y_true,y_pred',
                 output='acc.',
                 description='todo',
                 n_classes=26 + 1,
                 epochs=2,
                 save_projected_representations=True,
                 save_raw_representations=True):
        super().__init__(idx, name, input_, output, description)
        self.n_epochs = epochs
        self.n_classes = n_classes
        self.save_projected_representations = save_projected_representations
        self.save_raw_representations = save_raw_representations

        self.pbar = None
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, ssl_model, dataloader_train, dataloader_test):
        assert dataloader_train.num_gpus == 1
        steps_train = len(dataloader_train.dataset
                          ) / dataloader_train.batch_size * self.n_epochs
        steps_test = len(dataloader_test.dataset) / dataloader_test.batch_size
        self.pbar = tqdm(total=int(steps_train + steps_test))
        self.pbar.set_description(
            "Training and Evaluating on downstream task".center(50))
        self._evaluate(ssl_model, dataloader_train, dataloader_test)
        self.pbar.close()

    def _evaluate(self, ssl_model, dataloader_train, dataloader_test):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """
        representation_dim = 512
        if representation_dim == 0:
            raise NotImplementedError("Representation dim could not be inferred")

        ssl_model.normalize = False
        ssl_model.projector = nn.Linear(representation_dim,self.n_classes).to(self.device)
        ssl_model = ssl_model.to(self.device)
        ssl_model.projector.weight.data.normal_(mean=0.0, std=0.01)
        ssl_model.projector.bias.data.zero_()
        self._train_transfer_task(ssl_model, dataloader_train, dataloader_test)
        self._test_transfer_task(ssl_model, dataloader_test)

    def forward_adapter(self, model, x):
        representation = model._forward_backbone(x)
        return representation
     
    def _train_transfer_task(self, ssl_model, dataloader, dataloader_test):
        """Starts training for the downstream task
        """
        writer = SummaryWriter()
        if self.n_epochs == 0:
            return
        loss_fc = nn.CrossEntropyLoss()
        optimizer = optim.SGD(lr=0.01, momentum=0.9,weight_decay=1e-6,params=ssl_model.projector.parameters())
        ssl_model.eval()

        for epoch in range(self.n_epochs):
            loss_record = AverageMeter()
            dl = dataloader(epoch)
            bi = None
            for batch_idx, (x1, _, y, idx) in enumerate(dl,start=epoch * len(dl)):
                x1 = x1.to(self.device).float()
                y = y.view(-1).to(self.device).long()
                with torch.no_grad():
                    representation = self.forward_adapter(ssl_model, x1)
                assert representation.requires_grad == False
                pred = ssl_model.projector(representation)
                loss = loss_fc(pred, y)
                loss_record.update(loss.item(), pred.size(0))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.pbar.update(1)
                self.pbar.set_description("Loss {} Epoch {}".format(str(loss_record.avg)[:5], epoch +1))
                writer.add_scalar("Train Loss", loss.item(),
                                  batch_idx)
                bi = batch_idx
            logging.info(
                "Downstream task training epoch {}/{} loss: {}".format(
                    epoch + 1, self.n_epochs, loss_record.avg))
            # calc test set loss
            losses = []
            if True:
                for (x1, _, y, idx) in dataloader_test(epoch):
                    x1 = x1.to(self.device).float()
                    y = y.view(-1).to(self.device).long()
                    with torch.no_grad():
                        representation = self.forward_adapter(ssl_model, x1)
                        pred = ssl_model.projector(representation)
                        loss = loss_fc(pred, y)
                        losses.append(loss.item())
                writer.add_scalar("Test Loss", sum(losses) / len(losses), bi)

    def _test_transfer_task(self, ssl_model, dataloader):
        """Evaluates accuracy on a previously trained linear classifier
        """
        correct = 0
        n_samples = 0
        ssl_model.eval()
        labels = torch.tensor([])
        preds = torch.tensor([]).to(self.device)
        for batch_idx, (x1, _, y, idx) in enumerate(dataloader()):
            with torch.no_grad():
                representation = self.forward_adapter(ssl_model, x1.to(self.device).float())
                pred = ssl_model.projector(representation)
                labels = torch.cat([labels, y])
                pred = torch.argmax(pred, dim=1)
                preds = torch.cat([preds, pred])
                correct += pred.eq(y.to(
                    self.device).view_as(pred)).sum().item()
                n_samples += pred.shape[0]
                self.pbar.update(1)
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy())
        print(report)
        acc = correct / len(dataloader.dataset[0])
        print("Downstream task accuracy {}".format(acc))
        logging.info("Downstream task test accuracy: {}".format(acc))
        return acc
