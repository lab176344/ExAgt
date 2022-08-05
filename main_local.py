from experiment import Experiment
from torchvision import  transforms


meta_info = {"name":"HAHA", "description":"description123"}

# train number
# train hypers
training = {'idx':0,'traintest':[70,30], "num_gpus":1}
# dataset number
# dataset hypers
dataset_train = [{'name':'argoverse','augmentation_type':["no"],'mode':'train','bbox_meter': [200, 200],'bbox_pixel':[200,200],'center_meter':[100.0,100.0],\
           'hist_seq_first':0,'hist_seq_last':49,'representation_type':'image','orientation':'north'},{'name':'argoverse','augmentation_type':["connectivity"],'mode':'train','bbox_meter': [200, 200],'bbox_pixel':[200,200],'center_meter':[100.0,100.0],\
           'hist_seq_first':0,'hist_seq_last':49,'representation_type':'image','orientation':'north'}]
                   

dataset_test = [{'name':'argoverse','augmentation_type':["no"],'mode':'train','bbox_meter': [200, 200],'bbox_pixel':[200,200],'center_meter':[100.0,100.0],\
           'hist_seq_first':0,'hist_seq_last':49,'representation_type':'image','orientation':'north'}]

# dataloader number
# dataloader hypers

train_dataloader = {'idx':0, 'batch_size':2,'epochs':100, 'num_workers':0,'shuffle':True,
                    'representation':'trajectory','grid_chosen':[0,3,6,9]}              
       


test_dataloader = {'idx':0, 'batch_size':2, 'num_workers':0,
                   'shuffle':False,'representation':'trajectory','grid_chosen':[0,3,6,9]}

# model number
# model hypers

model = {'idx':3,'model_depth':18, 'projector_dim': [1024, 2048]}
#model = {'idx':100,'encoderI_type':"ResNet-18",'encoderT_type':"Transformer-Encoder",'merge_type':"FC"}

# eval number
# eval hypers
evaluation = {'idx':0}

# optimis er numer
# optimiser hypers
optimiser = {'idx':0,'lr':0.001,'weight_decay':0,'betas':(0.9, 0.999)}

# scheduler number
# scheduler hypers
scheduler = None

# loss number
# loass hypers
loss = {'idx':4}


if __name__ == '__main__':
    experiment = Experiment(meta_info,dataset_train,dataset_test,train_dataloader,test_dataloader,model,training,evaluation,optimiser,scheduler,loss)
    #experiment.save_experiment_config(add_timestamp=True)
    #experiment = Experiment.from_config_file("./saved_configs/name123_09-04-2021_13-52-33.pkl")
    experiment.train()
    experiment.evaluate()
