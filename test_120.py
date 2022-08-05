from src.dataset.dataset import dataset
from src.dataloader.dataloader_120 import dataloader_120
import matplotlib.pyplot as plt
import time
import numpy
from tqdm import tqdm
import numpy as np
from scipy.io import savemat
from src.model.model_120 import generate_model
import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.utils.average_meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import io
import cv2
from torchvision.transforms import functional as TF


representation_type = 'image_multichannel_vector'
dataset_test_dict = {'name':'argoverse','augmentation_type': None, 'augmentation_meta': {'range':25.0,'angle_range':30.0}, 'hist_seq_first': 0,
                     'hist_seq_last':20,'pred_seq_last':50,'orientation':'ego','representation_type':'image_multichannel_vector','mode':'train',
                     'bbox_meter':[60.0,60.0],'bbox_pixel':[240,240],'center_meter':[20.0,30.0],'only_edges':False}
dataset_local = [dataset(**dataset_test_dict)]
loader = dataloader_120(dataset=dataset_local,batch_size=256,shuffle=False,num_workers=8)
loader = loader()
model_dict = {'encoderI_type':'ResNet-18','encoderT_type':'LSTM','merge_type':'Transformer', 'decoder_type':'LSTM','z_dim_t':64,
              'z_dim_i':64,'zm_dim_in':64,'zm_dim_out':128,'image_size':240,'channels':5,'traj_size':3,
            'm_depth':6, 'm_heads':8}
model = generate_model(**model_dict)

# for idx,data in enumerate(tqdm(dataset_local[0])):
#     channel_to_show = 0
#     image = data['image'][channel_to_show,:,:]
#     center = dataset_test_dict['center_meter']
#     resolution = numpy.array(dataset_test_dict['bbox_pixel'])/numpy.array(dataset_test_dict['bbox_meter'])
#     plt.imshow(image[:,:])
#     plt.axis('off')
#     aa = data['traj_hist_obj']
    
#     for i in range(7):
#         print('X:',max(aa[i,:,0]),'Y:',max(aa[i,:,1]))
    
#     for idTraj,(traj,seq_len) in enumerate(zip(data['traj_hist_obj'],data['traj_hist_obj_seq_lens'])):
#         traj = traj.t()
#         if idTraj == (data['traj_hist_obj'].shape[0]-1):
#             plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(traj[1,:seq_len]+center[1])*resolution[1],'b')
#         else:
#             plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(traj[1,:seq_len]+center[1])*resolution[1],'g')
            
#     # for idTraj,(traj,seq_len) in enumerate(zip(data['traj_pred_obj'],data['traj_pred_obj_seq_lens'])):
#     #     traj = traj.t()
#     #     if idTraj == (data['traj_pred_obj'].shape[0]-1):
#     #         plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(traj[1,:seq_len]+center[1])*resolution[1],'bo')
#     #     else:
#     #         plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(traj[1,:seq_len]+center[1])*resolution[1],'go')
            
#     plt.show()
#     break



data_storage = False
data_visualize = True


if data_storage:
    collectImage = []
    collectTraj = []
    collectObj = []
    collectMaskObj = []
    collectLen = []

    for idx,data in enumerate(tqdm(dataset_local[0])):
        image = data['image'][0,:,:]
        traj = data['traj_ego']
        obj = data['traj_objs']
        mask_obbj = data['mask_objs']
        len_obj = data['traj_objs_lens']
        traj = traj.reshape(1,traj.shape[0],traj.shape[1])
        collectImage.append(image.cpu().numpy().reshape(1,80,80))
        collectTraj.append(traj)
    collectImage = np.concatenate(collectImage,axis=0)
    collectTraj = np.concatenate(collectTraj,axis=0)
    savemat('collectImage.mat',{'image':collectImage,'traj':collectTraj})
    savemat('collectImageObj.mat',{'Obj':collectObj,'maskObj':collectMaskObj,'ObjLen':collectLen})

if data_visualize:
    for idx in [119835]:
        #idx = 1023 # 30 125 131 136 212 213 229
        print(idx)
        dataset_local_plot = dataset_local[0]
        if representation_type=='image':
            image = dataset_local_plot[idx]['image'][0,:,:,:]
            for i in range(image.shape[0]):
                plt.imshow(image[i,:,:])
                plt.pause(0.01)
                plt.axis('off')
                #plt.savefig('images_collection/image_aug1_'+str(i)+'.pdf',bbox_inches='tight')
                plt.cla()  
        elif representation_type=='image_vector':
            image = dataset_local_plot[idx]['image'][0,:,:]
            traj =dataset_local_plot[idx]['traj_ego'][:2,:]
            mask = dataset_local_plot[idx]['mask_ego']
            center = dataset_local_plot.center
            resolution = numpy.array(dataset_local_plot.bbox_pixel)/numpy.array(dataset_local_plot.bbox_meter)
            plt.imshow(image[:,:])
            plt.axis('off')
            plt.plot((traj[0,:]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,:]+center[1])*resolution[1],'b')
            plt.plot((traj[0,mask]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,mask]+center[1])*resolution[1],'bo')
            for traj,mask,seq_len in zip(dataset_local_plot[idx]['traj_objs'],dataset_local_plot[idx]['mask_objs'],dataset_local_plot[idx]['traj_objs_lens']):
                traj = traj.t()
                plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,:seq_len]+center[1])*resolution[1],'g')
                plt.plot((traj[0,:seq_len][mask]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,:seq_len][mask]+center[1])*resolution[1],'go')
            plt.savefig('traj.png')
            plt.show()
            
        elif representation_type=='image_multichannel_vector':
            channel_to_show = 0
            image = dataset_local_plot[idx]['image'][channel_to_show,:,:]
            #mask = dataset_local[idx]['mask_ego']
            center = dataset_local_plot.center
            resolution = numpy.array(dataset_local_plot.bbox_pixel)/numpy.array(dataset_local_plot.bbox_meter)
            plt.imshow(image[:,:])
            plt.axis('off')
            for idTraj,(traj,seq_len) in enumerate(zip(dataset_local_plot[idx]['traj_hist_obj'],dataset_local_plot[idx]['traj_hist_obj_seq_lens'])):
                traj = traj.t()
                if idTraj == (dataset_local_plot[idx]['traj_hist_obj'].shape[0]-1):
                    plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,:seq_len]+center[1])*resolution[1],'b')
                else:
                    plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,:seq_len]+center[1])*resolution[1],'g')
                    
            for idTraj,(traj,seq_len) in enumerate(zip(dataset_local_plot[idx]['traj_pred_obj'],dataset_local_plot[idx]['traj_pred_obj_seq_lens'])):
                traj = traj.t()
                if idTraj == (dataset_local_plot[idx]['traj_pred_obj'].shape[0]-1):
                    plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,:seq_len]+center[1])*resolution[1],'bo')
                else:
                    plt.plot((traj[0,:seq_len]+center[0])*resolution[0],dataset_local_plot.bbox_pixel[1]-(traj[1,:seq_len]+center[1])*resolution[1],'go')
                    
            plt.savefig('traj.png')
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



for epc in range(1):
    writer = SummaryWriter()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_average = AverageMeter()
    for batch_idx_test in enumerate(tqdm(loader)):
        optimiser.zero_grad()
        x_image = batch_idx_test[1]['images']
        x_image = x_image.cuda()
        x_traj = [batch_idx_test[1]['hist_objs'].cuda(),batch_idx_test[1]['hist_obj_lens']]
        x_traj_len = batch_idx_test[1]['hist_objs_seq_len']
        x_traj_pred_len = batch_idx_test[1]['pred_objs_seq_len']
        batch_wise_objects_length_sum = batch_idx_test[1]['hist_object_lengths_sum']
        batch_wise_decoder_input = batch_idx_test[1]['obj_decoder_in'].cuda()
        gTruthX = batch_idx_test[1]['pred_objsx'].cuda()
        gTruthY = batch_idx_test[1]['pred_objsy'].cuda()
        
        
        target_len = 29
        model = model.to('cuda')
        aa = model(x_image=x_image,x_traj=x_traj,x_traj_len=x_traj_len,batch_wise_object_lengths_sum=batch_wise_objects_length_sum,
                batch_wise_decoder_input=batch_wise_decoder_input,target_length=target_len)
        X = aa['X']
        Y = aa['Y']
        obj_length_padded = torch.cat([torch.Tensor([0]),batch_wise_objects_length_sum]).int()

        X_reshaped = torch.empty_like(gTruthX,device=gTruthX.device)
        Y_reshaped = torch.empty_like(gTruthY,device=gTruthX.device)

        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp,0:x_traj[1][unp],:,:] = X[obj_length_padded[unp]:obj_length_padded[unp+1],:].unsqueeze(2)  
            Y_reshaped[unp,0:x_traj[1][unp],:,:] = Y[obj_length_padded[unp]:obj_length_padded[unp+1],:].unsqueeze(2)    

        maskX = gTruthX!=-100000
        maskY = gTruthY!=-100000

        X_mse = F.mse_loss(X_reshaped[maskX],gTruthX[maskX])
        Y_mse = F.mse_loss(Y_reshaped[maskY],gTruthY[maskY])
        
        loss = X_mse + Y_mse
        loss.backward()
        optimiser.step()
        loss_average.update(loss.item(),X.shape[0])
        # with torch.no_grad():
        #     if batch_idx_test[0]%20==0:
        #         resolution = numpy.array(dataset_test_dict['bbox_pixel'])/numpy.array(dataset_test_dict['bbox_meter'])
        #         seq_len = dataset_test_dict['hist_seq_last']
        #         center = dataset_test_dict['center_meter']
        #         for id,image in enumerate(x_image):
                    
        #             image = image[0,:,:].cpu().numpy()
        #             traj_hist = x_traj[0][id,:,:,:].cpu().numpy()
        #             gTruthX_plot = gTruthX[id,:,:,:].cpu().numpy()
        #             gTruthY_plot = gTruthY[id,:,:,:].cpu().numpy()
        #             X_reshaped_plot = X_reshaped[id,:,:,:].cpu().numpy()
        #             Y_reshaped_plot = Y_reshaped[id,:,:,:].cpu().numpy()
        #             image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        #             #plt.imshow(image)
        #             #plt.axis('off')
        #             for id,(x,y,x_temp_hist,y_temp_hist,gt_x,gt_y) in \
        #             enumerate(zip(X_reshaped_plot[obj_length_padded[id]:obj_length_padded[id+1],:,:],
        #                           Y_reshaped_plot[obj_length_padded[id]:obj_length_padded[id+1],:,:],
        #                           traj_hist[obj_length_padded[id]:obj_length_padded[id+1],:,0],
        #                           traj_hist[obj_length_padded[id]:obj_length_padded[id+1],:,1],
        #                           gTruthX_plot[obj_length_padded[id]:obj_length_padded[id+1],:,:],
        #                           gTruthY_plot[obj_length_padded[id]:obj_length_padded[id+1],:,:])):

                    
  
        #                 x = x[:x_traj_len[id],:]
        #                 y = y[:x_traj_len[id],:] 

        #                 x_temp_hist = x_temp_hist[:x_traj_len[id],]
        #                 y_temp_hist = y_temp_hist[:x_traj_len[id],]
                        
        #                 gt_x = gt_x[:x_traj_len[id],:]
        #                 gt_y = gt_y[:x_traj_len[id],:]
                             
        #                 line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((gt_x+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(gt_y+center[1])*resolution[1])]
        #                 line_cor = np.array(line_cor)
        #                 image = cv2.polylines(image, [line_cor], False, (0,255,0), 1)                   
        #                 #plt.plot((x+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(y+center[1])*resolution[1],'b')
        #                 #plt.plot((x_temp_hist+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(y_temp_hist+center[1])*resolution[1],'r')
        #                 #plt.plot((gt_x+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(gt_y+center[1])*resolution[1],'g')

        #                 line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((x_temp_hist+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(y_temp_hist+center[1])*resolution[1])]
        #                 line_cor = np.array(line_cor)
        #                 image = cv2.polylines(image, [line_cor], False, (255,0,0), 3)                   
                    
                    
        #                 line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((x+center[0])*resolution[0],dataset_test_dict['bbox_pixel'][1]-(y+center[1])*resolution[1])]
        #                 line_cor = np.array(line_cor)
        #                 image = cv2.polylines(image, [line_cor], False, (0,0,255), 1)
        #                 if id>10:
        #                     break
                        
        #                 image_to_write = TF.to_tensor(image)
                        #writer.add_image(('image_'+str(id)),image_to_write,epc)

    writer.add_scalar("Train Loss - AVG", loss_average.avg, epc)

    print('Epoch: ',epc,'Loss: ',loss_average.avg)
    
    



# python -m cProfile -o script.profile test.py
#pyprof2calltree -i script.profile -o script.calltree !!!!!!!!!!!!!!!!!! vprof -c h test.py
# %%
