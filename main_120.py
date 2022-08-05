from experiment import Experiment
from torchvision import  transforms


meta_info = {"name":"DriverBeTa_WorkingSetupDelayedTimeConstantLoss", "description":"Driver behaviour in the traffic"}


# dataset number
# dataset hypers
dataset_train = [{'name':'argoverse','augmentation_type': None, 'augmentation_meta': {'range':25.0,'angle_range':30.0}, 'hist_seq_first': 0,
                     'hist_seq_last':20,'pred_seq_last':50,'orientation':'ego','representation_type':'image_multichannel_vector','mode':'train',
                     'bbox_meter':[60.0,60.0],'bbox_pixel':[240,240],'center_meter':[20.0,30.0],'only_edges':False}]
                   


dataset_val = [{'name':'argoverse','augmentation_type': None, 'augmentation_meta': {'range':25.0,'angle_range':30.0}, 'hist_seq_first': 0,
                     'hist_seq_last':20,'pred_seq_last':50,'orientation':'ego','representation_type':'image_multichannel_vector','mode':'val',
                     'bbox_meter':[60.0,60.0],'bbox_pixel':[240,240],'center_meter':[20.0,30.0],'only_edges':False}]


# The test dataset ground truth is black box and only available via the evaluation server
dataset_test = [{'name':'argoverse','augmentation_type': None, 'augmentation_meta': {'range':25.0,'angle_range':30.0}, 'hist_seq_first': 0,
                     'hist_seq_last':20,'pred_seq_last':50,'orientation':'ego','representation_type':'image_multichannel_vector','mode':'val',
                     'bbox_meter':[60.0,60.0],'bbox_pixel':[240,240],'center_meter':[20.0,30.0],'only_edges':False}]

# train number
# train hypers
training = {'idx':120, "num_gpus":1, 'dataset_dict': dataset_train[0]}

# dataloader number
# dataloader hypers

train_dataloader = {'idx':120, 'batch_size':218,'epochs':100, 'num_workers':10,'shuffle':True,
                    'representation':'image_multichannel_vector'}              
       


val_dataloader = {'idx':120, 'batch_size':64, 'num_workers':15,
                   'shuffle':True,'representation':'image_multichannel_vector'}


test_dataloader = {'idx':120, 'batch_size':64, 'num_workers':15,
                   'shuffle':False,'representation':'image_multichannel_vector'}
# model number
# model hypers

#model = {'idx':0,'model_depth':18, 'projector_dim': [1024, 2048]}
model = {'idx':120,'encoderI_type':'ResNet-18','encoderT_type':'Transformer-Encoder','merge_type':'Transformer', 'decoder_type':'LSTM','z_dim_t':64,
              'z_dim_i':64,'zm_dim_in':64,'zm_dim_out':128,'image_size':240,'channels':1,'traj_size':3,
            'm_depth':6, 'm_heads':8}

# eval number
# eval hypers
evaluation = {'idx':120,'visualize':True}

# optimis er numer
# optimiser hypers
optimiser = {'idx':1,'lr':0.001,'weight_decay':0,'betas':(0.9, 0.999)}

# scheduler number
# scheduler hypers
scheduler = None

# loss number
# loass hypers
loss = [{'idx':120},{'idx':121,'thres_dynamic_1':8,'thres_dynamic_2':8}]


## Main file settings
settings_main = {'use_meta_info':False,'train':False}

if __name__ == '__main__':
    experiment = Experiment(meta_info,dataset_train,dataset_val,dataset_test,train_dataloader,val_dataloader,test_dataloader,model,training,evaluation,optimiser,scheduler,loss)
    if settings_main['use_meta_info']:
      experiment = Experiment.from_config_file("./saved_configs/name123_09-04-2021_13-52-33.pkl")
    else:
      experiment.save_experiment_config(add_timestamp=True)
    if settings_main['train']:
      experiment.train()
      experiment.save_checkpoint()
      #experiment.evaluate()
    else:
      experiment.load_checkpoint()
      experiment.evaluate()
