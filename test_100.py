from src.dataset.dataset import dataset
from src.dataloader.dataloader_101 import dataloader_101
import matplotlib.pyplot as plt
import time
import numpy
representation_type = 'image_vector_merge_recon'
dataset_test_dict = {'name':'argoverse','augmentation_type':["no"],'mode':'test','bbox_meter': [200, 200],'bbox_pixel':[100,100],'center_meter':[100.0,100.0],\
           'hist_seq_first':0,'hist_seq_last':49,'representation_type':'image_vector_merge_recon','orientation':'north'}
dataset_local = [dataset(**dataset_test_dict)]
loader = dataloader_101(dataset=dataset_local,batch_size=64)
loader = loader()
for batch_idx_test,proj_data in enumerate(loader):
    quit()
for a,pp,pn in loader(0,rank=0):
    if representation_type=='image':
        for i_b in range(a[0].shape[0]):
            image = a['image'][i_b,0,:,:,:]
            for i in range(image.shape[0]):
                plt.imshow(image[i,:,:])
                plt.pause(0.1)
                plt.cla()

            image = pp[0][i_b,0,:,:,:]
            for i in range(image.shape[0]):
                plt.imshow(image[i,:,:])
                plt.pause(0.1)
                plt.cla()
            
            image = pn[0][i_b,0,:,:,:]
            for i in range(image.shape[0]):
                plt.imshow(image[i,:,:])
                plt.pause(0.1)
                plt.cla()
    elif representation_type=='image_vector':
        for i_b in range(len(a[0])):
            fig, axs = plt.subplots(3)
            image = a[0][i_b,:,:]
            traj =  a[3][i_b,:2,:]
            center = dataset_local.center
            resolution = numpy.array(dataset_local.bbox_pixel)/numpy.array(dataset_local.bbox_meter)
            axs[0].imshow(image[:,:])
            axs[0].set_title(str(a[-1][i_b]))
            axs[0].plot((traj[0,:]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,:]+center[1])*resolution[1],'b')
            
            image = pp[0][i_b,:,:]
            traj =  pp[3][i_b,:2,:]
            center = dataset_local.center
            resolution = numpy.array(dataset_local.bbox_pixel)/numpy.array(dataset_local.bbox_meter)
            axs[1].imshow(image[:,:])
            axs[1].set_title(str(pp[-1][i_b]))
            axs[1].plot((traj[0,:]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,:]+center[1])*resolution[1],'b')
            
            image = pn[0][i_b,:,:]
            traj =  pn[3][i_b,:2,:]
            center = dataset_local.center
            resolution = numpy.array(dataset_local.bbox_pixel)/numpy.array(dataset_local.bbox_meter)
            axs[2].imshow(image[:,:])
            axs[2].set_title(str(pn[-1][i_b]))
            axs[2].plot((traj[0,:]+center[0])*resolution[0],dataset_local.bbox_pixel[1]-(traj[1,:]+center[1])*resolution[1],'b')
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
    quit()

#python -m cProfile -o script.profile test.py
#pyprof2calltree -i script.profile -o script.calltree !!!!!!!!!!!!!!!!!! vprof -c h test.py