import logging

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from src.evaluation.eval import eval
from src.model.model_3 import BasicBlock, Bottleneck, model_3
from src.model.model_5 import model_5
from src.utils.average_meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# from sklearn.metrics import top_k_accuracy_score

class eval_7(eval):
    def __init__(self,
                 idx=7,
                 name='supervised training',
                 input_='y_true,y_pred',
                 output='acc.',
                 description='todo',
                 n_classes=26 + 1,
                 epochs=25):
        super().__init__(idx, name, input_, output, description)
        self.n_epochs = epochs
        self.n_classes = n_classes
        self.pbar = None
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def forward_adapter(self, model, x):
        if isinstance(model, model_3):
            representation = model._forward_backbone(x)
            return representation
        if isinstance(model, model_5):
            representation = model.backbone_and_projector._forward_backbone(x)
            representation = nn.functional.normalize(representation, dim=1, p=2)
            pred = model.projector(representation)
            return pred


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

        ssl_model.normalize = False
        ssl_model.projector = nn.Linear(representation_dim, self.n_classes).to(
            self.device)
        ssl_model = ssl_model.to(self.device)

        self._train_transfer_task(ssl_model, dataloader_train, dataloader_test)
        self._test_transfer_task(ssl_model, dataloader_test)

    def _train_transfer_task(self, ssl_model, dataloader, dataloader_test):
        """Starts training for the downstream task
        """
        loss_fc = nn.CrossEntropyLoss()
        optimizer = optim.Adam(ssl_model.parameters(), lr=0.001)
        ssl_model.train()

        writer = SummaryWriter()

        for epoch in range(self.n_epochs):
            loss_record = AverageMeter()
            dl = dataloader(epoch)
            bi = None
            for batch_idx, (x1, _, y, idx) in enumerate(dl,start=epoch * len(dataloader)):

                x1 = x1.to(self.device).float()
                y = y.view(-1).to(self.device).long()
                representation = ssl_model._forward_backbone(
                    x1.to(self.device).float())
                pred = ssl_model._forward_projector(representation)
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
            if True:
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

        save_path = f"./checkpoints/supervised_model_1p.pth"
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
                representation = ssl_model._forward_backbone(
                    x1.to(self.device).float())
                pred = ssl_model._forward_projector(representation)
                pred = torch.argmax(pred, dim=1)
                preds = torch.cat([preds, pred])
                correct += pred.eq(y.to(
                    self.device).view_as(pred)).sum().item()
                n_samples += pred.shape[0]
                self.pbar.update(1)
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy())
        # top5  = top_k_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy(), k=5)
        # print(f"Top 5 {top5}")
        acc = correct / len(dataloader.dataset[0])
        print(report)
        print("Downstream task accuracy {}".format(acc))
        logging.info("Downstream task test accuracy: {}".format(acc))
        return acc
