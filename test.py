from src.dataset.dataset import dataset
from src.dataloader.dataloader_100 import dataloader_100
import matplotlib.pyplot as plt
import time
import numpy
representation_type = 'image_vector'
dataset_test_dict = {'name':'openTraffic','augmentation_type': None, 'augmentation_meta': {'range':25.0,'angle_range':30.0}, 'hist_seq_first': 0,
                     'hist_seq_last':20,'pred_seq_last':50,'orientation':'ego','representation_type':representation_type,'mode':'train',
                     'bbox_meter':[60.0,60.0],'bbox_pixel':[240,240],'center_meter':[20.0,30.0],'only_edges':False}
dataset_local = dataset(**dataset_test_dict)
idx = 2
if representation_type=='image':
    image = dataset_local[idx]['image'][0,:,:,:]
    for i in range(image.shape[0]):
        #plt.title(int(dataset_local[idx]['label'][0]))
        plt.imshow(image[i,:,:])
        plt.pause(0.1)
        plt.cla()
elif representation_type=='image_vector':
    image = dataset_local[idx]['image'][0,:,:]
    traj =dataset_local[idx]['traj_ego'][:2,:]
    mask = dataset_local[idx]['mask_ego']
    center = dataset_local.center
    resolution = numpy.array(dataset_local.bbox_pixel)/numpy.array(dataset_local.bbox_meter)
    plt.imshow(image[:,:])
    plt.plot((traj[0,:]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,:]+center[1])*resolution[1],'b')
    plt.plot((traj[0,mask]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,mask]+center[1])*resolution[1],'bo')
    for traj,mask,seq_len in zip(dataset_local[idx]['traj_objs'],dataset_local[idx]['mask_objs'],dataset_local[idx]['traj_objs_lens']):
        traj = traj.t()
        plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,:seq_len]+center[1])*resolution[1],'g')
        plt.plot((traj[0,:seq_len][mask]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,:seq_len][mask]+center[1])*resolution[1],'go')
    plt.show()
elif representation_type=='vector':
    for line,mask in zip(dataset_local[idx]['map'],dataset_local[idx]['mask_map']):
        plt.plot(line[0,:],line[1,:],'r')
        plt.plot(line[0,mask],line[1,mask],'ro')
    for obj,mask in zip(dataset_local[idx]['traj_objs'],dataset_local[idx]['mask_objs']):
        plt.plot(obj[0,:],obj[1,:],'g')
        plt.plot(obj[0,mask],obj[1,mask],'go')
    traj =dataset_local[idx]['traj_ego'][:2,:]
    mask = dataset_local[idx]['mask_ego']
    plt.plot(traj[0,:],traj[1,:],'b')
    plt.plot(traj[0,mask],traj[1,mask],'bo')

    plt.show()


#python -m cProfile -o script.profile test.py
#pyprof2calltree -i script.profile -o script.calltree !!!!!!!!!!!!!!!!!! vprof -c h test.py