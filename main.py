from experiment import Experiment
from torchvision import transforms
import torchio
meta_info = {"name": "argonewnew_baseline_double_trans",
             "description": "baseline + connectivity for new argo data with barlow"}

# train number
# train hypers
training = {'idx': 0, 'traintest': [70, 30], "num_gpus": 4}
# dataset number
# dataset hypers
dataset_name = "argonew"
dataset_train = [{'name': dataset_name, 'mode': 'train',
                  'augmentation_type': {"connectivity": 0.3, "fieldofview": 0.7},
                  'bbox_meter': [60, 60], 'bbox_pixel': [120, 120],
                  'center_meter': [20.0, 30.0], \
                  'hist_seq_first': 0, 'hist_seq_last': 50, 'pred_seq_first': 0,
                  'pred_seq_last': 0, 'orientation': 'ego'},  # x1

                 {'name': dataset_name, 'mode': 'train', 'bbox_meter': [60, 60],
                  'augmentation_type': {"connectivity": 0.7, "fieldofview": 0.3},
                  'bbox_pixel': [120, 120], 'center_meter': [20.0, 30.0], \
                  'hist_seq_first': 0, 'hist_seq_last': 50, 'pred_seq_first': 0,
                  'pred_seq_last': 0, 'orientation': 'ego'}]  # x2

dataset_test = [{'name': dataset_name, 'mode': 'val', 'bbox_meter': [60, 60],
                 'bbox_pixel': [120, 120], 'center_meter': [20.0, 30.0], \
                 'hist_seq_first': 0, 'hist_seq_last': 50, 'pred_seq_first': 0,
                 'pred_seq_last': 0, 'orientation': 'ego'},
                {'name': dataset_name, 'mode': 'val', 'bbox_meter': [60, 60],
                 'bbox_pixel': [120, 120], 'center_meter': [20.0, 30.0], \
                 'hist_seq_first': 0, 'hist_seq_last': 50, 'pred_seq_first': 0,
                 'pred_seq_last': 0, 'orientation': 'ego'}]

# dataloader number
# dataloader hypers

trans_train_x1 = transforms.Compose([
    transforms.RandomCrop(80),
    transforms.RandomRotation(degrees=(-10, 10), fill=(0,), center=(20, 30)),
    transforms.RandomApply(transforms=[transforms.GaussianBlur(5)], p=0.3),
    transforms.RandomApply(
        transforms=[torchio.transforms.RandomNoise(std=(0, 0.1))], p=0.7),
])

trans_train_x2 = transforms.Compose([
    transforms.RandomCrop(80),
    transforms.RandomRotation(degrees=(-10, 10), fill=(0,), center=(20, 30)),
    transforms.RandomApply(transforms=[transforms.GaussianBlur(5)], p=0.7),
    transforms.RandomApply(
        transforms=[torchio.transforms.RandomNoise(std=(0, 0.1))], p=0.3),
])

train_dataloader = {'idx': 0, 'batch_size': 64, 'epochs': 30, 'num_workers': 4,
                    'shuffle': True,
                    'representation': 'image',
                    'transformation': [trans_train_x1, trans_train_x2],
                    'grid_chosen': [0, 3, 6, 9]}

test_dataloader = {'idx': 0, 'batch_size': 128, 'num_workers': 10,
                   'shuffle': False,
                   'representation': 'image', 'transformation': [trans_train_x2, None],
                   'grid_chosen': [0, 3, 6, 9]}

# model number
# model hypers

# model = {'idx':0,'model_depth':18, 'projector_dim': [1024, 2048]}
model = {'idx': 3, 'model_depth': 50, 'projector_dim': [2048, 2048, 2048]}

# eval number
# eval hypers
evaluation = {'idx': 2}

# optimiser numer
# optimiser hypers
optimiser = {'idx': 1, 'lr': 0.001, 'weight_decay': 0, 'betas': (0.9, 0.999)}

# scheduler number
# scheduler hypers
scheduler = None  # {'idx':0, 'base_lr': 0.06, 'final_lr':0.0006 , 'warmup_epochs': 0}

# loss number
# loass hypers
loss = {'idx': 0}

if __name__ == '__main__':
    models = ["model_epoch_15_base_double_resnet18_bs50.pth", "model_epoch_15_base_expert_double_resnet18_bs50.pth"]
    for mod in models:
        print(mod)
        experiment = Experiment(meta_info, dataset_train, dataset_test,
                                train_dataloader, test_dataloader, model, training,
                                evaluation, optimiser, scheduler, loss)
        experiment.save_experiment_config(add_timestamp=True)
        experiment.load_checkpoint("./checkpoints/" + mod)
        experiment.evaluate()
    # experiment.save_checkpoint(meta_info["name"])
    # experiment.load_checkpoint("./checkpoints/" + meta_info["name"] + ".pth")
    # experiment.evaluate()

