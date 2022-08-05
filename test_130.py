from src.dataset.dataset import dataset
from src.dataloader.dataloader_130 import dataloader_130
import matplotlib.pyplot as plt
import time
import numpy
representation_type = 'trajectory'
dataset_local = dataset(name='argoverse',augmentation_type=["no"],augmentation_meta={'range':50.0,'angle_range':120.0},hist_seq_first=0,hist_seq_last=19,pred_seq_last=49,orientation='north',representation_type=representation_type,mode='train',bbox_meter=[100.0,100.0],bbox_pixel=[200,200],center_meter=[50.0,50.0])
loader = dataloader_130(dataset=[dataset_local],batch_size=64)

for sample in loader(0,rank=0):
    for i_b in range(sample['hist_objs'].shape[0]):
        for i_obj in range(sample['hist_obj_lens'][i_b]):
            traj_hist =  sample['hist_objs'][i_b,i_obj,:sample['hist_objs_seq_len'][i_b][i_obj],:2]
            plt.plot(traj_hist[:,0],traj_hist[:,1],'g')
            traj_pred =  sample['pred_objs'][i_b,i_obj,:sample['pred_objs_seq_len'][i_b][i_obj],:2]
            plt.plot(traj_pred[:,0],traj_pred[:,1],'m')
        
        
        traj_hist =  sample['hist_ego'][i_b,:,:2]
        plt.plot(traj_hist[:,0],traj_hist[:,1],'b')
        traj_pred =  sample['pred_ego'][i_b,:,:2]
        plt.plot(traj_pred[:,0],traj_pred[:,1],'r')
        plt.show()
   


#python -m cProfile -o script.profile test.py
#pyprof2calltree -i script.profile -o script.calltree !!!!!!!!!!!!!!!!!! vprof -c h test.py