import logging

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from src.utils.average_meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class eval_transfer_learning(object):
    def __init__(self,
                 idx=5,
                 name='Transfer task finetuning',
                 input_='y_true,y_pred',
                 output='acc.',
                 description='todo',
                 n_classes=26 + 1,
                 epochs=15):
        super().__init__(idx, name, input_, output, description)
        self.n_epochs = epochs
        self.n_classes = n_classes
        self.pbar = None
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def forward_adapter(self, model, x):
        representation = model._forward_backbone(x)
        return representation

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
        representation_dim = 512
        if representation_dim == 0:
            raise NotImplementedError(
                "Representation dim could not be inferred")

        ssl_model.normalize = False
        ssl_model.projector = nn.Linear(representation_dim, self.n_classes).to(
            self.device)
        ssl_model = ssl_model.to(self.device)
        ssl_model.projector.weight.data.normal_(mean=0.0, std=0.01)
        ssl_model.projector.bias.data.zero_()
        self._train_transfer_task(ssl_model, dataloader_train, dataloader_test)
        self._test_transfer_task(ssl_model, dataloader_test)

    def _train_transfer_task(self, ssl_model, dataloader, dataloader_test):
        """Starts training for the downstream task
        """
        classifier_parameters, model_parameters = [], []
        for name, param in ssl_model.named_parameters():
            if name in {"projector.weight","projector.bias"}:
                classifier_parameters.append(param)
            else:
                model_parameters.append(param)


        param_groups = [{
            "params": classifier_parameters, "lr":0.001,
        },{
            "params": model_parameters, "lr": 0.0001
        }]
        loss_fc = nn.CrossEntropyLoss()
        optimizer = optim.Adam(param_groups, 0)
        ssl_model.train()

        writer = SummaryWriter()

        for epoch in range(self.n_epochs):
            loss_record = AverageMeter()
            dl = dataloader(epoch)
            bi = None
            for batch_idx, (x1, _, y, idx) in enumerate(dl,start=epoch * len(dataloader)):

                x1 = x1.to(self.device).float()
                y = y.view(-1).to(self.device).long()
                pred = self.forward_adapter(ssl_model, x1)
                loss = loss_fc(pred, y)
                loss_record.update(loss.item(), pred.size(0))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.pbar.update(1)
                self.pbar.set_description("Loss {} Epoch {}".format(str(loss_record.avg)[:5],epoch +1))
                writer.add_scalar("Train Loss", loss.item(),
                                  batch_idx)
                bi = batch_idx
            logging.info(
                "Downstream task training epoch {}/{} loss: {}".format(
                    epoch + 1, self.n_epochs, loss_record.avg))
            # calc test set loss
            losses = []
            if False:
                for (x1, _, y, idx) in dataloader_test(epoch):
                    x1 = x1.to(self.device).float()
                    y = y.view(-1).to(self.device).long()
                    with torch.no_grad():
                        representation = ssl_model._forward_backbone(
                            x1.to(self.device).float())
                        pred = ssl_model._forward_projector(representation)
                        loss = loss_fc(pred, y)
                        losses.append(loss.item())
                writer.add_scalar("Test Loss", sum(losses) / len(losses), bi)

        save_path = f"./checkpoints/finetuned_model.pth"
        torch.save(ssl_model.state_dict(), save_path)

    def _test_transfer_task(self, ssl_model, dataloader):
        """Evaluates accuracy on a previously trained linear classifier
        """
        print("Start evaluation")
        correct = 0
        n_samples = 0
        ssl_model.eval()
        labels = torch.tensor([])
        preds = torch.tensor([]).to(self.device)
        for batch_idx, (x1, _, y, idx) in enumerate(dataloader()):
            with torch.no_grad():
                labels = torch.cat([labels, y])
                pred = self.forward_adapter(ssl_model, x1.to(self.device).float())
                pred = torch.argmax(pred, dim=1)
                preds = torch.cat([preds, pred])
                correct += pred.eq(y.to(
                    self.device).view_as(pred)).sum().item()
                n_samples += pred.shape[0]
                self.pbar.update(1)
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy())

        acc = correct / len(dataloader.dataset[0])
        print(report)
        print("Downstream task accuracy {}".format(acc))
        logging.info("Downstream task test accuracy: {}".format(acc))
        return acc
