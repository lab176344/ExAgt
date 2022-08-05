import logging
import os
import pickle
from datetime import datetime
import numpy as np
import torch

from src.dataset.dataset import dataset
from src.utils.average_meter import AverageMeter

logging.basicConfig(filename="./logs.log",
                            filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

class Experiment(object):
    def __init__(self,
                 meta_info = None,
                 dataset_param_train = None,
                 dataset_param_val = None,
                 dataset_param_test = None,
                 trdataloader_param = None,
                 vadataloader_param = None,
                 tedataloader_param = None,
                 model_param = None,
                 training_param = None,
                 evaluation_param = None,
                 optimiser_param = None,
                 scheduler_param = None,
                 loss_param = None) -> None:
        
        super().__init__()
        logging.info("Launching new experiment {}".format(meta_info))
        self.meta_info = meta_info
        self.dataset_param_train = dataset_param_train
        self.dataset_param_val = dataset_param_val
        self.dataset_param_test = dataset_param_test
        self.trdataloader_param = trdataloader_param
        self.vadataloader_param = vadataloader_param
        self.tedataloader_param = tedataloader_param
        self.model_param = model_param
        self.training_param = training_param
        self.evaluation_param = evaluation_param
        self.optimiser_param = optimiser_param
        self.scheduler_param = scheduler_param
        self.loss_param = loss_param

        self.trdataloader_param = {**self.trdataloader_param, "num_gpus":self.training_param["num_gpus"]}
        # Dataset and Dataloader
        self.dataset_train = []
        self.dataset_val = []
        self.dataset_test = []
        for i in range(len(self.dataset_param_train)):
            self.dataset_train.append(dataset(**self.dataset_param_train[i]))

        for i in range(len(self.dataset_param_val)):
            self.dataset_val.append(dataset(**self.dataset_param_val[i]))

        for i in range(len(self.dataset_param_test)):
            self.dataset_test.append(dataset(**self.dataset_param_test[i]))
        logging.info(str(self.dataset_train[0]))
        logging.info(str(self.dataset_val[0]))
        logging.info(str(self.dataset_test[0]))

        dataloader_name = 'dataloader_' + str(trdataloader_param['idx'])
        import_str = "from src.dataloader.{0} import {0}".format(dataloader_name)
        exec(import_str) 
        self.dataloader_train = eval(dataloader_name + '(dataset=self.dataset_train,**self.trdataloader_param)')
        self.dataloader_val = eval(dataloader_name + '(dataset=self.dataset_val,**self.vadataloader_param)')
        self.dataloader_test = eval(dataloader_name + '(dataset=self.dataset_test,**self.tedataloader_param)')
        # Model
        model_name = 'model_' + str(model_param['idx'])
        import_str = "from src.model.{0} import generate_model".format(model_name)
        exec(import_str) 
        self.model = eval('generate_model(**self.model_param)')        
        self.best_loss = 0
        # optimiser
        optimiser_name = 'optimizer_' + str(optimiser_param['idx'])
        import_str = "from src.train.optimizer.{0} import {0}".format(optimiser_name)
        exec(import_str) 
        self.optimiser = eval(optimiser_name+'(params = list(self.model.parameters()),**self.optimiser_param)')
        # Scheduler; Optional
        if scheduler_param is not None:
            scheduler_name = 'scheduler_' + str(scheduler_param['idx'])
            import_str = "from src.train.scheduler.{0} import {0}".format(
                scheduler_name)
            exec(import_str)
            dataloader_len = len(self.dataloader_train)
            epochs = self.dataloader_train.epochs
            self.scheduler_param = {
                **self.scheduler_param, 'dataloader_len': dataloader_len,
                'epochs': epochs
            }
            self.scheduler = eval(scheduler_name + '(**self.scheduler_param)')
        else:
            self.scheduler = None

        # losss
        if type(loss_param) is list:
            self.loss = [] # empty list for all the loss functions
            for loss_paral_local in loss_param:
                loss_name = 'loss_' + str(loss_paral_local['idx'])
                import_str = "from src.train.loss.{0} import {0}".format(loss_name)
                exec(import_str) 
                self.loss.append(eval(loss_name+'(**loss_paral_local)'))
        else:
            loss_name = 'loss_' + str(loss_param['idx'])
            import_str = "from src.train.loss.{0} import {0}".format(loss_name)
            exec(import_str) 
            self.loss = eval(loss_name+'(**self.loss_param)')
        # eval
        eval_name = 'eval_' + str(evaluation_param['idx'])
        import_str = "from src.evaluation.{0} import {0}".format(eval_name)
        exec(import_str)
        self.eval_task = eval(eval_name+'(**self.evaluation_param)')
        self.epochs = self.trdataloader_param['epochs']
        # train
        train_name = 'train_' + str(training_param['idx'])
        import_str = "from src.train.{0} import {0}".format(train_name)
        exec(import_str)
        self.training_task = eval(train_name + '(**self.training_param)')

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'

    def __str__(self):
        task = "TASK: " + self.model.task if self.model.task != None else ""
        name = "NAME: " + self.model.name if self.model.name != None else ""
        desc = "DESCRIPTION: " + self.model.description if self.model.description != None else ""
        return name +" "  + task + " " +desc


    def save_checkpoint(self):
        path_str = os.getcwd()
        path_str = os.path.join(path_str,"checkpoints")
        if not os.path.exists(path_str):
            os.makedirs(path_str)
        path_str = os.path.join(path_str, self.meta_info["name"]+ ".pth")       
        try:
            torch.save(self.model.state_dict(), path_str)
        except Exception as e:
            print("Saving model failed", e)

    def load_checkpoint(self):
        path_str = os.getcwd()
        path_str = os.path.join(path_str,"checkpoints")
        path_str = os.path.join(path_str, self.meta_info["name"]+ ".pth")       
        state_dict = torch.load(path_str)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #    name = k[7:]  # remove `module.`
        #    new_state_dict[name] = v
        # load params
        self.model.load_state_dict(state_dict)

    def train(self):
        logging.info("Started training for experiment {}".format(str(self)))
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        logging.info("Total model params: {:.0f}M (trainable: {:.0f}M)".format(
            total_params / 1e6, trainable_params / 1e6))

        self.training_task.run_training(self.model, self.dataloader_train,
                                 self.dataloader_val, self.loss, self.optimiser, self.scheduler, self.save_checkpoint)


    def evaluate(self):
        self.eval_task(self.model, self.dataloader_test,None,self.dataset_param_train)


    def save_experiment_config(self,
                               path: str = None,
                               add_timestamp: bool = False):
        """Saves experiment config as pickle file to disk

        Args:
            path: Optional; path to which the config is saved. If no path is
                specified the filename will be the name specified in the
                meta_information
            add_timestamp: Optional; Whether or not to append a timestamp to the
                filename
        Note:
            Does not save model params
        """
        config = {
            "meta_info": self.meta_info,
            "dataset_param_train": self.dataset_param_train,
            "dataset_param_val": self.dataset_param_val,
            "trdataloader_param": self.trdataloader_param,
            "tedataloader_param": self.tedataloader_param,
            "model_param": self.model_param,
            "training_param": self.training_param,
            "evaluation_param": self.evaluation_param,
            "optimiser_param": self.optimiser_param,
            "scheduler_param": self.scheduler_param,
            "loss_param": self.loss_param
        }
        if not path:
            os.makedirs("./saved_configs/", exist_ok=True)
            path = "./saved_configs/" + self.meta_info["name"]
        if add_timestamp:
            timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            path += "_{}".format(timestamp)
        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(config, f)
        logging.info("Saved config {} \n to file {}".format(config, path))

    @classmethod
    def from_config_file(cls, path: str):
        """Factory for creating an Experiment with a previously saved config

        Args:
            path: path to config

        Returns:
            Experiment
        """
        with open(path, 'rb') as f:
            return cls(**pickle.load(f))
