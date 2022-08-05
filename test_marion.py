from src.dataset.dataset import dataset
import matplotlib.pyplot as plt
import time
import numpy
representation_type = 'graph'
start = time.time()
dataset_local = dataset(augmentation=False,hist_seq_first=0,hist_seq_last=49,pred_seq_last=100,orientation='ego',name='openTraffic',representation_type=representation_type,mode='train',bbox_meter=[100.0,100.0],bbox_pixel=[200,200],center_meter=[50.0,50.0])
end = time.time()
print(end - start)
print(len(dataset_local))

idx = 0
if representation_type=='image':
    image = dataset_local[idx]['image'][0,:,:,:]
    for i in range(image.shape[0]):
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
    for traj,mask in zip(dataset_local[idx]['traj_objs'],dataset_local[idx]['mask_objs']):
        plt.plot((traj[0,:]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,:]+center[1])*resolution[1],'g')
        plt.plot((traj[0,mask]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,mask]+center[1])*resolution[1],'go')
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
elif representation_type=='graph':
    F_X = dataset_local[idx]['F_X']
    F_Y = dataset_local[idx]['F_Y']
    F_VX = dataset_local[idx]['F_VX']
    F_VY = dataset_local[idx]['F_VY']
    XY_pred = dataset_local[idx]['XY_pred']


#python -m cProfile -o script.profile test.py
#pyprof2calltree -i script.profile -o script.calltree !!!!!!!!!!!!!!!!!! vprof -c h test.py
