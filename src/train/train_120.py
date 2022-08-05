import logging
from datetime import datetime
from einops import rearrange

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.train.train import Training
from src.utils.average_meter import AverageMeter
from src.evaluation.eval_120 import eval_120    
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.get_fde import get_fde
from src.utils.get_ade import get_ade
from src.utils.get_img_from_fig import get_img_from_fig
from src.utils.get_cmap import get_cmap

class train_120(Training):
    def __init__(self, idx, **kwargs) -> None:
        super().__init__()
        self.description = "Driver Behaviour Model training logic"
        self.dataset_dict = kwargs['dataset_dict']

    def run_training(self, model, dataloader_train, dataloader_test, loss_fc, optimizer,
                     scheduler,save_checkpoint):
        if dataloader_train.num_gpus > 1:
            mp.spawn(self._train_dist,
                     nprocs=dataloader_train.num_gpus,
                     args=(model, dataloader_train, dataloader_test, loss_fc, optimizer,self.dataset_dict,save_checkpoint))
        else:
            self._train(model, dataloader_train, dataloader_test , loss_fc, optimizer,self.dataset_dict,save_checkpoint)
    @staticmethod
    def _train(model, dataloader_train, dataloader_test, loss_fc, optimizer,dataset_dict,save_checkpoint) -> None:
        """Single gpu training
        """
        epochs = dataloader_train.epochs
        writer = SummaryWriter()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        time_start = datetime.now()
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        model.to(device)
        opimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        eval_func = eval_120()
        if type(loss_fc) is list:
            mse_loss = loss_fc[0]
            delta_loss = loss_fc[1]


        for epoch in range(epochs):
            model.train()

            loss_record = AverageMeter()
            ade_record = AverageMeter()
            fde_record = AverageMeter()
            T_mse_record = AverageMeter()
            max_dy_1 = AverageMeter()
            max_dy_2 = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch)
            for batch_idx, sample in enumerate(dataloader,
                                              start=epoch * len(dataloader)):
                opimizer.zero_grad()
                x_image = sample['images'].to(device, non_blocking=True)
                x_traj = [sample['hist_objs'].to(device,non_blocking=True),sample['hist_obj_lens']]
                x_traj_len = sample['hist_objs_seq_len']
                x_traj_pred_obj_len = sample['pred_obj_lens']
                x_traj_pred_len = sample['pred_objs_seq_len']
                obj_length_padded = sample['hist_object_lengths_sum']
                pres_object_lengths_sum = sample['pres_object_lengths_sum']
                batch_wise_decoder_input = sample['obj_decoder_in'].to(device, non_blocking=True)
                gTruthX = sample['pred_objsx'].to(device, non_blocking=True)
                gTruthY = sample['pred_objsy'].to(device, non_blocking=True)
                gTruthT = sample['pred_objst'].to(device, non_blocking=True)
                target_len = 29 # trajectory length as a parameter should be included in the training
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
                X_mse = mse_loss(X_reshaped[~maskXnot],gTruthX[~maskXnot])
                Y_mse = mse_loss(Y_reshaped[~maskYnot],gTruthY[~maskYnot])
                T_mse = mse_loss(T_reshaped[~maskXnot],gTruthT[~maskYnot])
                delta_loss_val,max_delta_1,max_delta_2 = delta_loss(dynamic_1,dynamic_2)
                loss = X_mse + Y_mse + delta_loss_val + T_mse
                loss.backward()
                with torch.no_grad():
                    ade = get_ade(X_reshaped, Y_reshaped, gTruthX, gTruthY, maskXnot, maskYnot)
                    fde = get_fde(X_reshaped, Y_reshaped, gTruthX, gTruthY, x_traj_pred_len, maskXnot, maskYnot)
                opimizer.step()

                fde_record.update(fde.item(), x_image.size(0))
                T_mse_record.update(T_mse.item(), x_image.size(0))
                ade_record.update(ade.item(), x_image.size(0))
                loss_record.update(loss.item(),x_image.shape[0])
                max_dy_1.update(torch.max(max_delta_1).item(),x_image.shape[0])
                max_dy_2.update(torch.max(max_delta_2).item(),x_image.shape[0])
                del X_mse
                del Y_mse
                del loss
                del output
                del fde
                del ade
                #del delta_loss_val
                del max_delta_1
                del max_delta_2
                with torch.no_grad():
                    if batch_pass%200==0:
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

                            #fig_dy_x = plt.figure()
                            #fig_dy_y = plt.figure()
                            #dy_x_ax1 = fig_dy_x.add_subplot(111)
                            #dy_y_ax1 = fig_dy_y.add_subplot(111)

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
                                if dy_x.shape[0]!=1 and dy_y.shape[0]!=1:
                                    dy_x = np.diff(dy_x.squeeze())/0.1
                                    dy_y = np.diff(dy_y.squeeze())/0.1
                                    dy_x = np.expand_dims(dy_x,axis=1)
                                    dy_y = np.expand_dims(dy_y,axis=1)
                                    #dy_x_ax1.plot(list(range(len(gt_x)-1)),dy_x,color=cmap_objec(id),linewidth=1)
                                    #dy_y_ax1.plot(list(range(len(gt_y)-1)),dy_y,color=cmap_objec(id),linewidth=1)
      
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
                            #dy_x_image = get_img_from_fig(fig_dy_x)
                            #dy_x_image = cv2.resize(dy_x_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)  
                            #dy_y_image = get_img_from_fig(fig_dy_y)
                            #dy_y_image = cv2.resize(dy_y_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)
                            #dy_x_image = dy_x_image
                            #dy_y_image = dy_y_image
                            # clear the figure
                            #plt.close(fig_dy_x)
                            #dy_x_ax1.cla()
                            #plt.close(fig_dy_y)
                            #dy_y_ax1.cla()

                            image_to_display = image
                            #image_to_display = np.concatenate((image,dy_x_image,dy_y_image),axis=1)
                             
                            writer.add_image(('image_'+str(idx)),image_to_display,batch_pass,dataformats='HWC')
                            del image_to_display
                            if idx>10:
                                break
                                

                batch_pass = batch_pass + 1
                pbar.update(1)
                log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:.3f} ADE: {:.3f} FDE: {:.3f} D1: {:.3f}  D2: {:.3f} T_mse: {:.3f}".format(
                    epoch + 1, epochs, batch_pass, len(dataloader),
                    round(loss_record.avg, 3), ade_record.avg, fde_record.avg, max_dy_1.avg,max_dy_2.avg,T_mse_record.avg).center(50)
                pbar.set_description(log_msg)
                writer.add_scalar("Train BatchWise Loss", loss_record.avg, batch_idx)
                logging.info(log_msg)
            
            writer.add_scalar("Train Loss", loss_record.avg, epoch)
            writer.add_scalar("FDE", fde_record.avg, epoch)
            writer.add_scalar("ADE", ade_record.avg, epoch)
            writer.add_scalar("Max_dy_1", max_dy_1.avg, epoch)
            writer.add_scalar("Max_dy_2", max_dy_2.avg, epoch)

            print('\nEpoch: {}/{} Train Loss: {:.3f}'.format(epoch + 1, epochs, loss_record.avg))
            if epoch%5==0:
                eval_loss,ade_eval,fde_eval = eval_func(model,dataloader_test,device)
                print('\nEpoch: {}/{} Test Loss: {:.3f}'.format(epoch + 1, epochs, eval_loss))
                save_checkpoint()
                writer.add_scalar("Test Loss", eval_loss, epoch)
                writer.add_scalar("Test ADE", ade_eval, epoch)
                writer.add_scalar("Test FDE", fde_eval, epoch)


        pbar.set_description("Training finished {} (Total time: {})".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            datetime.now() - time_start).center(50))

        pbar.close()
        logging.info(
            "End of Training -- total time: {}".format(datetime.now() -
                                                       time_start))

    @staticmethod
    def _train_dist(rank, model, dataloader_train, dataloader_test, loss_fc, optimizer,dataset_dict,save_checkpoint) -> None:
        assert dist.is_nccl_available()
        #TODO merge with _train
        writer = SummaryWriter()
        world_size = dataloader_train.num_gpus
        epochs = dataloader_train.epochs
        eval_func = eval_120()
        if type(loss_fc) is list:
            mse_loss = loss_fc[0]
            delta_loss = loss_fc[1]
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              (dataloader_train.batch_size)),
                    desc="init training...".center(50))

        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=world_size,
                                rank=rank)

        device = torch.device(rank)
        model.to(device)
        if dataloader_train.batch_size < 4:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model, broadcast_buffers=False, device_ids=[rank],find_unused_parameters=True)
        for epoch in range(epochs):
            model.train()

            loss_record = AverageMeter()
            fde_record = AverageMeter()
            ade_record = AverageMeter()
            T_mse_record = AverageMeter()
            max_dy_1 = AverageMeter()
            max_dy_2 = AverageMeter()
            batch_pass = 0
            dataloader = dataloader_train(epoch, rank)
            for batch_idx, sample in enumerate(dataloader,
                                              start=epoch * len(dataloader)):
                optimizer.zero_grad()
                x_image = sample['images'].to(device, non_blocking=True)
                x_traj = [sample['hist_objs'].to(device,non_blocking=True),sample['hist_obj_lens']]
                x_traj_len = sample['hist_objs_seq_len']
                x_traj_pred_obj_len = sample['pred_obj_lens']
                x_traj_pred_len = sample['pred_objs_seq_len']
                obj_length_padded = sample['hist_object_lengths_sum']
                pres_object_lengths_sum = sample['pres_object_lengths_sum']
                batch_wise_decoder_input = sample['obj_decoder_in'].to(device, non_blocking=True)
                gTruthX = sample['pred_objsx'].to(device, non_blocking=True)
                gTruthY = sample['pred_objsy'].to(device, non_blocking=True)
                gTruthT = sample['pred_objst'].to(device, non_blocking=True)
                target_len = 29 # trajectory length as a parameter should be included in the training
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
                X_mse = mse_loss(X_reshaped[~maskXnot],gTruthX[~maskXnot])
                Y_mse = mse_loss(Y_reshaped[~maskYnot],gTruthY[~maskYnot])
                T_mse = mse_loss(T_reshaped[~maskXnot],gTruthT[~maskYnot])
                delta_loss_val,max_delta_1,max_delta_2 = delta_loss(dynamic_1,dynamic_2)
                loss = X_mse + Y_mse + T_mse + delta_loss_val
                loss.backward()
                with torch.no_grad():
                    ade = get_ade(X_reshaped, Y_reshaped, gTruthX, gTruthY, maskXnot, maskYnot)
                    fde = get_fde(X_reshaped, Y_reshaped, gTruthX, gTruthY, x_traj_pred_len, maskXnot, maskYnot)
                optimizer.step()
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                dist.all_reduce(ade, dist.ReduceOp.SUM)
                dist.all_reduce(fde, dist.ReduceOp.SUM)
                dist.all_reduce(T_mse, dist.ReduceOp.SUM)
                loss_record.update(loss.item() / world_size,
                                   x_image.size(0) * world_size)
                fde_record.update(fde.item() / world_size,
                                   x_image.size(0) * world_size)
                ade_record.update(ade.item() / world_size,
                                   x_image.size(0) * world_size)
                T_mse_record.update(T_mse.item()/ world_size, 
                                    x_image.size(0)* world_size)


                max_dy_1.update(torch.max(max_delta_1).item(),x_image.shape[0])
                max_dy_2.update(torch.max(max_delta_2).item(),x_image.shape[0])
                del X_mse
                del Y_mse
                del loss
                del output
                del fde
                del ade
                #del delta_loss_val
                del max_delta_1
                del max_delta_2
                with torch.no_grad():
                    if batch_pass%200==0 and rank==0:
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

                            #fig_dy_x = plt.figure()
                            #fig_dy_y = plt.figure()
                            #dy_x_ax1 = fig_dy_x.add_subplot(111)
                            #dy_y_ax1 = fig_dy_y.add_subplot(111)

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
                                if dy_x.shape[0]!=1 and dy_y.shape[0]!=1:
                                    dy_x = np.diff(dy_x.squeeze())/0.1
                                    dy_y = np.diff(dy_y.squeeze())/0.1
                                    dy_x = np.expand_dims(dy_x,axis=1)
                                    dy_y = np.expand_dims(dy_y,axis=1)
                                    #dy_x_ax1.plot(list(range(len(gt_x)-1)),dy_x,color=cmap_objec(id),linewidth=1)
                                    #dy_y_ax1.plot(list(range(len(gt_y)-1)),dy_y,color=cmap_objec(id),linewidth=1)
      
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
                            #dy_x_image = get_img_from_fig(fig_dy_x)
                            #dy_x_image = cv2.resize(dy_x_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)  
                            #dy_y_image = get_img_from_fig(fig_dy_y)
                            #dy_y_image = cv2.resize(dy_y_image, (image.shape[0],image.shape[1]), interpolation = cv2.INTER_AREA)
                            #dy_x_image = dy_x_image
                            #dy_y_image = dy_y_image
                            # clear the figure
                            #plt.close(fig_dy_x)
                            #dy_x_ax1.cla()
                            #plt.close(fig_dy_y)
                            #dy_y_ax1.cla()

                            image_to_display = image
                            #image_to_display = np.concatenate((image,dy_x_image,dy_y_image),axis=1)
                             
                            writer.add_image(('image_'+str(idx)),image_to_display,batch_pass,dataformats='HWC')
                            del image_to_display
                            if idx>10:
                                break
                                

                batch_pass = batch_pass + 1
                if rank==0:
                    pbar.update(world_size)
                    log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:.3f} ADE: {:.3f} FDE: {:.3f} D1: {:.3f}  D2: {:.3f} T_mse: {:.3f}".format(
                        epoch + 1, epochs, batch_pass*world_size, len(dataloader)*world_size,
                        round(loss_record.avg, 3), ade_record.avg, fde_record.avg, max_dy_1.avg,max_dy_2.avg,T_mse_record.avg).center(50)
                    pbar.set_description(log_msg)
                    writer.add_scalar("Train BatchWise Loss", loss_record.avg, batch_idx)
                    logging.info(log_msg)
            if rank==0:
                writer.add_scalar("Train Loss", loss_record.avg, epoch)
                writer.add_scalar("FDE", fde_record.avg, epoch)
                writer.add_scalar("ADE", ade_record.avg, epoch)
                writer.add_scalar("Max_dy_1", max_dy_1.avg, epoch)
                writer.add_scalar("Max_dy_2", max_dy_2.avg, epoch)

                print('\nEpoch: {}/{} Train Loss: {:.3f}'.format(epoch + 1, epochs, loss_record.avg))
            if epoch%5==0 and rank==0:
                eval_loss,ade_eval,fde_eval = eval_func(model,dataloader_test,device)
                print('\nEpoch: {}/{} Test Loss: {:.3f}'.format(epoch + 1, epochs, eval_loss))
                save_checkpoint()
                writer.add_scalar("Test Loss", eval_loss, epoch)
                writer.add_scalar("Test ADE", ade_eval, epoch)
                writer.add_scalar("Test FDE", fde_eval, epoch) 
