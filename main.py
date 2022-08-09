from torchvision import transforms
import torchio
from src.dataloader import dataloader
from src.dataset import dataset
from src.model_cross_view import model_cross_view
from src.loss_barlow_twins import loss as loss_barlow_twins
from src.loss_vic_reg import loss as loss_vic_reg
import torch.optim as optim
from src.train import train
from src.eval_clustering_accuracy import eval_clustering_accuracy
from src.eval_linear_classifier import eval_linear_classifier
from src.eval_transfer_learning import eval_transfer_learning
import torch
import os

'''
Data Preperation
'''


dataset_name = 'argoverse'

dataset_train_args_1 = {'name': dataset_name, 'mode': 'train',
                  'augmentation_type': {"connectivity": 0.3, "fieldofview": 0.7},
                  'bbox_meter': [60, 60], 'bbox_pixel': [120, 120],
                  'center_meter': [20.0, 30.0], \
                  'seq_first': 0, 'seq_last': 50, 'orientation': 'ego'} 
dataset_train_args_2 = {'name': dataset_name, 'mode': 'train', 'bbox_meter': [60, 60],
                  'augmentation_type': {"connectivity": 0.7, "fieldofview": 0.3},
                  'bbox_pixel': [120, 120], 'center_meter': [20.0, 30.0], \
                  'seq_first': 0, 'seq_last': 50, 'orientation': 'ego'}

dataset_train = list()
dataset_train.append(dataset(**dataset_train_args_1))
dataset_train.append(dataset(**dataset_train_args_2))


dataset_test_args = {'name': dataset_name, 'mode': 'val', 'bbox_meter': [60, 60],
                 'bbox_pixel': [120, 120], 'center_meter': [20.0, 30.0], \
                 'seq_first': 0, 'seq_last': 50, 'orientation': 'ego'}

dataset_test = [dataset(**dataset_test_args)]


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

train_dataloader_args = {'batch_size': 64, 'epochs': 30, 'num_workers': 4,
                    'shuffle': True,
                    'transformation': [trans_train_x1, trans_train_x2],
                    'grid_chosen': [0, 3, 6, 9]}

test_dataloader_args = {'batch_size': 128, 'num_workers': 10,
                   'shuffle': False,
                   'transformation': [trans_train_x2, None],
                   'grid_chosen': [0, 3, 6, 9]}

train_dataloader = dataloader(dataset=dataset_train,**train_dataloader_args)
test_dataloader = dataloader(dataset=dataset_test,**test_dataloader_args)


'''
Training (+ model init etc)
'''

loss_type = 'barlow_twins' # 'vic_reg'
eval_type = 'clustering_accuracy' # 'linear_classifier' # 'transfer_learning'

model_args = {'projector_dim': [2048, 2048, 2048]}
model = model_cross_view(**model_args)

optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=0, betas=(0.9, 0.999))

if loss_type == 'barlow_twins':
    loss = loss_barlow_twins()
elif loss_type == 'vic_reg':
    loss = loss_vic_reg()
    
train_obj = train() 

#os.system('tensorboard --logdir=./runs')

train_obj.run_training(model, train_dataloader, loss, optimiser)

eval_linear_args = {'n_classes':27,'epochs':2}
eval_transfer_learning_args = {'n_classes':27,'epochs':15}


if eval_type == 'clustering_accuracy':
    eval_task = eval_clustering_accuracy()
elif eval_type == 'linear_classifier':
    eval_task = eval_linear_classifier(**eval_linear_args)
elif eval_type == 'transfer_learning':
    eval_task = eval_transfer_learning(**eval_transfer_learning_args)
    
eval_task(model,train_dataloader,test_dataloader)
torch.save(model.state_dict(), 'model_cross_view.pt')




