import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.evaluation.eval import eval
from src.utils.average_meter import AverageMeter
from src.utils.get_fde import get_fde
from src.utils.get_ade import get_ade
import cv2
from src.utils.get_cmap import get_cmap
import logging
import matplotlib.pyplot as plt
from src.utils.get_img_from_fig import get_img_from_fig
import os
import imageio

class eval_120(eval):
    def __init__(self, 
                 idx=120,
                 name='MSE prediction accuracy',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=False,
                 description='Calculates MSE,ADE,FDE for all vehicles in the batch',
                  ):
        super().__init__(idx,
                    name,
                    input_,
                    output,
                    description)
        self.visualize = visualize

    def __call__(self, model=None, dataloader_test=None, device=None,dataset_dict=None):
        return self._evaluate(model, dataloader_test, device,dataset_dict)

    def _evaluate(self, model, dataloader_test, device,dataset_dict):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """
        def mse_eval(out_1 = None, out_2 = None):
            return F.mse_loss(out_1, out_2, reduction = 'mean')
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_record = AverageMeter()
        ade_record = AverageMeter()
        fde_record = AverageMeter()
        model.eval()
        if next(model.parameters()).device is not device:
            model.to(device)
        
        for batch_idx, sample in enumerate(dataloader_test(epoch=0)):
            x_image = sample['images']
            x_image = x_image.to(device)
            x_traj = [sample['hist_objs'].to(device),sample['hist_obj_lens']]
            x_traj_len = sample['hist_objs_seq_len']
            x_traj_pred_obj_len = sample['pred_obj_lens']
            x_traj_pred_len = sample['pred_objs_seq_len']
            pres_object_lengths_sum = sample['pres_object_lengths_sum']
            obj_length_padded = sample['hist_object_lengths_sum']
            batch_wise_decoder_input = sample['obj_decoder_in'].to(device)
            gTruthX = sample['pred_objsx'].to(device)
            gTruthY = sample['pred_objsy'].to(device)
            gTruthT = sample['pred_objst'].to(device, non_blocking=True)

            target_len = 29
            output = model(x_image=x_image,x_traj=x_traj,x_traj_len=x_traj_len,batch_wise_object_lengths_sum=obj_length_padded,
                    batch_wise_decoder_input=batch_wise_decoder_input,target_length=target_len)
            X = output['X']
            Y = output['Y']
            T = output['T']
            dynamic_1 = output['dynamic_1']
            dynamic_2 = output['dynamic_2']
            X_reshaped = torch.empty_like(gTruthX,device=gTruthX.device)
            Y_reshaped = torch.empty_like(gTruthY,device=gTruthX.device)
            T_reshaped = torch.empty_like(gTruthT,device=gTruthX.device)
            dynamic_1_reshaped = torch.empty_like(gTruthX,device=dynamic_1.device)
            dynamic_2_reshaped = torch.empty_like(gTruthY,device=dynamic_2.device)


            for unp in range(gTruthX.shape[0]):
                X_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = X[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)  
                Y_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = Y[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2) 
                dynamic_1_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = dynamic_1[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)
                dynamic_2_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = dynamic_2[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)
                T_reshaped[unp,0:x_traj_pred_obj_len[unp],:,:] = T[pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1],:].unsqueeze(2)


            maskXnot = torch.isnan(gTruthX)
            maskYnot = torch.isnan(gTruthX)
            X_mse = mse_eval(X_reshaped[~maskXnot],gTruthX[~maskXnot])
            Y_mse = mse_eval(Y_reshaped[~maskYnot],gTruthY[~maskYnot])
            T_mse = mse_eval(T_reshaped[~maskXnot],gTruthT[~maskXnot])
            loss = X_mse + Y_mse + T_mse
            ade = get_ade(X_reshaped, Y_reshaped, gTruthX, gTruthY, maskXnot, maskYnot)
            fde = get_fde(X_reshaped, Y_reshaped, gTruthX, gTruthY, x_traj_pred_len, maskXnot, maskYnot)
            ade_record.update(ade.item(), x_image.size(0))
            fde_record.update(fde.item(), x_image.size(0))
            loss_record.update(loss.item(), x_image.size(0))
            logging.info("Validation loss: {}".format(loss_record.avg))
            logging.info("Validation ADE: {}".format(ade_record.avg))
            logging.info("Validation FDE: {}".format(fde_record.avg))
            with torch.no_grad():
                if self.visualize:
                    save_path = os.path.join(os.getcwd(),'visImage')
                    save_path_gif = os.path.join(os.getcwd(),'visImage_gif')
                    if not os.path.exists(save_path_gif):
                        os.makedirs(save_path_gif)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if type(dataset_dict) is list:
                        dataset_dict = dataset_dict[0]
                    resolution = np.array(dataset_dict['bbox_pixel'])/np.array(dataset_dict['bbox_meter'])
                    seq_len = dataset_dict['hist_seq_last']
                    center = dataset_dict['center_meter']
                    for idx,image in enumerate(x_image):
                        
                        image = image[0,:,:].cpu().numpy()
                        traj_hist = x_traj[0][idx,:,:,:].cpu().numpy()
                        gTruthX_plot = gTruthX[idx,:,:,:].cpu().numpy()
                        gTruthY_plot = gTruthY[idx,:,:,:].cpu().numpy()
                        X_reshaped_plot = X_reshaped[idx,:,:,:].cpu().numpy()
                        Y_reshaped_plot = Y_reshaped[idx,:,:,:].cpu().numpy()
                        dynamic_1_plot = dynamic_1_reshaped[idx,:,:,:].cpu().numpy()
                        dynamic_2_plot = dynamic_2_reshaped[idx,:,:,:].cpu().numpy()
                        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

                        x_traj_len_local = x_traj_len[obj_length_padded[idx]:obj_length_padded[idx+1]]
                        x_traj_pred_len_local = x_traj_pred_len[obj_length_padded[idx]:obj_length_padded[idx+1]]
                        cmap_objec = get_cmap(X_reshaped_plot.shape[0])

                        fig_dy_x = plt.figure()
                        fig_dy_y = plt.figure()
                        dy_x_ax1 = fig_dy_x.add_subplot(111)
                        dy_y_ax1 = fig_dy_y.add_subplot(111)
                        dy_x_ax1.set_ylim([-4,4])
                        dy_y_ax1.set_ylim([-4,4])
                        #dy_x_ax1.legend()
                        #dy_y_ax1.legend()
                        dy_x_ax1.set_title('a_x')
                        dy_y_ax1.set_title('a_y')
                        for id,(x,y,x_temp_hist,y_temp_hist,gt_x,gt_y,dy_x,dy_y) in \
                        enumerate(zip(X_reshaped_plot[0:x_traj[1][idx],:,:],
                                    Y_reshaped_plot[0:x_traj[1][idx],:,:],
                                    traj_hist[0:x_traj[1][idx],:,0],
                                    traj_hist[0:x_traj[1][idx],:,1],
                                    gTruthX_plot[0:x_traj[1][idx],:,:],
                                    gTruthY_plot[0:x_traj[1][idx],:,:],
                                    dynamic_1_plot[0:x_traj[1][idx],:,:],
                                    dynamic_2_plot[0:x_traj[1][idx],:,:])):

                        

                            x = x[:x_traj_pred_len_local[id],:]
                            y = y[:x_traj_pred_len_local[id],:] 

                            x_temp_hist = x_temp_hist[:x_traj_len_local[id],]
                            y_temp_hist = y_temp_hist[:x_traj_len_local[id],]
                            
                            gt_x = gt_x[:x_traj_pred_len_local[id],:]
                            gt_y = gt_y[:x_traj_pred_len_local[id],:]

                            dy_x = dy_x[:x_traj_pred_len_local[id],:]
                            dy_y = dy_y[:x_traj_pred_len_local[id],:]
                            dy_x = dy_x[1:,:]
                            dy_y = dy_y[1:,:]
                            if dy_x.shape[0]!=1 and dy_y.shape[0]!=1:
                                dy_x = np.diff(dy_x.squeeze())/0.1
                                dy_y = np.diff(dy_y.squeeze())/0.1
                                dy_x = np.expand_dims(dy_x,axis=1)
                                dy_y = np.expand_dims(dy_y,axis=1)
                                dy_x_ax1.plot(list(range(len(gt_x)-2)),dy_x,color=cmap_objec(id),linewidth=1)
                                dy_y_ax1.plot(list(range(len(gt_y)-2)),dy_y,color=cmap_objec(id),linewidth=1)

                            line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((gt_x+center[0])*resolution[0],dataset_dict['bbox_pixel'][1]-(gt_y+center[1])*resolution[1])]
                            line_cor = np.array(line_cor)
                            line_cor = line_cor.reshape(-1,1,2)
                            image = cv2.polylines(image, [line_cor], False, (0,1,0), 1)                   

                            line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((x_temp_hist+center[0])*resolution[0],dataset_dict['bbox_pixel'][1]-(y_temp_hist+center[1])*resolution[1])]
                            line_cor = np.array(line_cor)
                            line_cor = line_cor.reshape(-1,1,2)

                            image = cv2.polylines(image, [line_cor], False, cmap_objec(id), 3)   


                        
                            line_cor = [(int(x_temp),int(y_temp)) for x_temp,y_temp in zip((x+center[0])*resolution[0],dataset_dict['bbox_pixel'][1]-(y+center[1])*resolution[1])]
                            line_cor = np.array(line_cor)
                            line_cor = line_cor.reshape(-1,1,2)

                            image = cv2.polylines(image, [line_cor], False, (0,0,1), 1)

                        #plt.show()
                        #image_to_display = rearrange(image,'H W C->C H W')
                        dy_x_image = get_img_from_fig(fig_dy_x)
                        dy_x_image = cv2.resize(dy_x_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)  
                        dy_y_image = get_img_from_fig(fig_dy_y)
                        dy_y_image = cv2.resize(dy_y_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)
                        dy_x_image = dy_x_image
                        dy_y_image = dy_y_image
                        # clear the figure
                        plt.close(fig_dy_x)
                        dy_x_ax1.cla()
                        plt.close(fig_dy_y)
                        dy_y_ax1.cla()

                        #image_to_display = image
                        image_to_display = np.concatenate((image*255.0,dy_x_image,dy_y_image),axis=1)
                        cv2.imwrite(os.path.join(save_path,str(batch_idx)+'_'+str(idx)+'.png'),image_to_display)
                        del image_to_display
                        if idx>5:
                            break

                    # GIF writer
                    for idx,image in enumerate(x_image):
                            
                        image = image[0,:,:].cpu().numpy()
                        traj_hist = x_traj[0][idx,:,:,:].cpu().numpy()
                        gTruthX_plot = gTruthX[idx,:,:,:].cpu().numpy()
                        gTruthY_plot = gTruthY[idx,:,:,:].cpu().numpy()
                        X_reshaped_plot = X_reshaped[idx,:,:,:].cpu().numpy()
                        Y_reshaped_plot = Y_reshaped[idx,:,:,:].cpu().numpy()

                        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

                        x_traj_len_local = x_traj_len[obj_length_padded[idx]:obj_length_padded[idx+1]]
                        x_traj_pred_len_local = x_traj_pred_len[obj_length_padded[idx]:obj_length_padded[idx+1]]

                        x_pred_temp = X_reshaped_plot[0:x_traj[1][idx],:,:]
                        y_pred_temp = Y_reshaped_plot[0:x_traj[1][idx],:,:]
                        x_hist_temp = traj_hist[:x_traj[1][idx],:,0]
                        y_hist_temp = traj_hist[:x_traj[1][idx],:,1]
                        gtruth_x_temp = gTruthX_plot[0:x_traj[1][idx],:,:]
                        gtruth_y_temp = gTruthY_plot[0:x_traj[1][idx],:,:]
                        frames = []
                        ímage_old = image.copy()
                        image_base = image.copy()
                        for id in range(x_hist_temp.shape[1]+x_pred_temp.shape[1]):
                            if id < x_hist_temp.shape[1]:
                                ímage_old = image.copy()

                                x_temp = (x_hist_temp[:,id] + center[0])*resolution[0]
                                y_temp = dataset_dict['bbox_pixel'][1]-(y_hist_temp[:,id]+center[1])*resolution[1]
                                for cid,(circle_x_temp,circle_y_temp) in enumerate(zip(x_temp.astype(int),y_temp.astype(int))):
                                    #alpha = 0.75
                                    #beta = 1 - alpha
                                    #image_draw = cv2.addWeighted(image_base,alpha,ímage_old,beta,0)
                                    cv2.circle(ímage_old,(circle_x_temp,circle_y_temp),3,cmap_objec(cid),-1)
                                frames.append(ímage_old)
                            else:
                                image_base_draw = image.copy()
                                cv2.putText(image_base_draw, 'GT', (220, 220), cv2.FONT_HERSHEY_PLAIN, 1, (1,0,0), 2)
                                cv2.putText(image_base_draw, 'Pred', (220, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,1,0), 2)

                                x_temp = (x_pred_temp[:,id-x_hist_temp.shape[1],:]+ center[0])*resolution[0]
                                y_temp = dataset_dict['bbox_pixel'][1]-(y_pred_temp[:,id-x_hist_temp.shape[1],:]+center[1])*resolution[1]
                                for circle_x_temp,circle_y_temp in zip(x_temp.astype(int),y_temp.astype(int)):
                                    if circle_x_temp[0]>0 and circle_y_temp[0]>0:
                                        cv2.circle(image_base_draw,(circle_x_temp[0],circle_y_temp[0]),3,(0,1,0),-1)
                                x_gTruth_temp = (gtruth_x_temp[:,id-x_hist_temp.shape[1],:]+ center[0])*resolution[0]
                                y_gTruth_temp = dataset_dict['bbox_pixel'][1]-(gtruth_y_temp[:,id-x_hist_temp.shape[1],:]+center[1])*resolution[1]
                                for circle_x_temp,circle_y_temp in zip(x_gTruth_temp.astype(int),y_gTruth_temp.astype(int)):
                                    if circle_x_temp[0]>0 and circle_y_temp[0]>0:
                                        cv2.circle(image_base_draw,(circle_x_temp[0],circle_y_temp[0]),3,(1,0,0),-1)
                                frames.append(image_base_draw)
                        save_gif_name = os.path.join(save_path_gif,str(batch_idx)+'_'+str(idx)+'.gif')
                        with imageio.get_writer(save_gif_name, mode="I") as writer:
                            for frame in frames:
                                writer.append_data(frame)
                        writer.close()
                        if idx>5:
                            break

        print("Validation loss: {}, Validation ADE: {}, Validation FDE: {}".format(loss_record.avg,ade_record.avg,fde_record.avg))
        return loss_record.avg, ade_record.avg, fde_record.avg
