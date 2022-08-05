import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F

class loss_100(loss):
    def __init__(self,
                idx = 100,
                name = 'Metric Loss + Weighted Recon Loss',
                description = 'Metric Loss given: anchor, pp, pn, nn (and optional similarity to pp) +  Weighted Recon Loss',
                input_ = '',
                output = '1D',
                ) -> None:
        super().__init__(idx,name,description,input_,output)

    def metric_loss(self,
                    z_a=None,
                    z_pp=None, 
                    z_pn=None, 
                    z_nn=None, 
                    d_in_pp=1.0, 
                    margin_nn=None,
                    margin_pn=None,
                    margin_pp=None, 
                    l2=None, 
                    beta_pp=None, 
                    beta_pn=None, 
                    beta_nn=None,
                    negatives_a=None,
                    negatives_pn=None,
                    negatives_pp=None,
                    negatives_nn=None,
                    hardest_sampling=True):

        d_pn = ((z_a - z_pn) ** 2).sum(dim=1)
        d_pp = ((z_a - z_pp) ** 2).sum(dim=1)
        d_pp_margin =  d_pp + F.relu(margin_pp-d_pp)
        z_nn_inside = torch.zeros_like(z_nn)
        for i in range(z_a.shape[0]):
            # Get all possible points and stack them
            z_nn_a = z_a[negatives_a[i,:],:]
            z_nn_pp = z_pp[negatives_pp[i,:],:]
            z_nn_pn = z_pn[negatives_pn[i,:],:]
            z_nn_nn = z_nn[negatives_nn[i,:],:]
            z_all_negatives = torch.cat((z_nn_a,z_nn_pp,z_nn_pn,z_nn_nn))
            d_nn_all = ((z_a[i,:] - z_all_negatives) ** 2).sum(dim=1)
            # Finding Semi-Hards
            mask = torch.gt(d_nn_all, d_pn[i]) & torch.lt( d_nn_all,d_pn[i] + margin_nn) # d_nn> d_pn & d_nn<d_pn+margin
            d_nn_all_sub = d_nn_all[mask]
            z_all_negatives_mask = z_all_negatives[mask,:]
            if len(d_nn_all_sub)>0:
                if hardest_sampling:
                    min_idx = torch.argmin(d_nn_all_sub)
                    z_nn_inside[i,:] = z_all_negatives_mask[min_idx,:]
                else:
                    min_idx = torch.randint(len(d_nn_all_sub),(1,))
                    z_nn_inside[i,:] = z_all_negatives_mask[min_idx,:]
            else:
                if hardest_sampling:
                    min_idx = torch.argmin(d_nn_all)
                    z_nn_inside[i,:] = z_all_negatives[min_idx,:]
                else:
                    min_idx = torch.randint(len(d_nn_all),(1,))
                    z_nn_inside[i,:] = z_all_negatives[min_idx,:]

        d_nn = ((z_a - z_nn_inside) ** 2).sum(dim=1)

        l_nn            = F.relu(d_pn                       + margin_nn - d_nn)  
        l_pn            = F.relu(d_pp_margin                + margin_pn - d_pn)
        l_pp            = (d_in_pp*margin_pp - d_pp)**2
        loss = beta_nn*l_nn + beta_pn*l_pn+ beta_pp*l_pp
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_a) + torch.norm(z_nn_inside) + torch.norm(z_pn) + beta_pp*torch.norm(z_pp))
        return loss

    def reconstruction_loss_weighted(self,
                                    x,x_pred,
                                    beta_traj=1.0,
                                    beta_infra=1.0,
                                    beta_back_infra=1.0,
                                    beta_back_traj=1.0):
        traj_base = torch.clone(x[:,0,:,:])
        mask_traj = traj_base>0.0
        traj_pred = torch.clone(x_pred[:,0,:,:])
        traj_pred = torch.mul(traj_pred,mask_traj)
        infra_base = torch.clone(x[:,1,:,:])
        mask_infra = infra_base>0.0
        infra_pred = torch.clone(x_pred[:,1,:,:])
        infra_pred = torch.mul(infra_pred,mask_infra)

        mask_back_traj = traj_base==0.0
        back_traj = torch.clone(x_pred[:,0,:,:])
        back_traj = torch.mul(back_traj,mask_back_traj)
        mask_back_infra = infra_base==0.0
        back_infra = torch.clone(x_pred[:,1,:,:])
        back_infra = torch.mul(back_infra,mask_back_infra)
        l_traj = ((traj_base - traj_pred) ** 2).sum()/mask_traj.sum()
        l_infra = ((infra_base - infra_pred) ** 2).sum()/mask_infra.sum()
        l_back_traj = (back_traj ** 2).sum()/mask_back_traj.sum()
        l_back_infra = (back_infra **2).sum()/mask_back_infra.sum()
        l = beta_traj * l_traj + beta_infra * l_infra + beta_back_traj * l_back_traj + beta_back_infra * l_back_infra
        return l
    
    def forward(self,
                tuplets, 
                margin_nn=1000.0,
                margin_pn=10.0,
                margin_pp=1.0, 
                l2=0.0, 
                beta_pp=0.0, 
                beta_pn=0.0, 
                beta_nn=1.0,
                alpha_recon=1.0, 
                alpha_metric=1.0, 
                beta_traj=1.0,
                beta_infra=1.0,
                beta_back_infra=1.0,
                beta_back_traj=1.0,
                hardest_sampling= True):
        """
        Computes loss for each batch.
        """
        z_a  = self.encode(tuplets['anchor_image'],tuplets['anchor_trajectory'])
        z_pp = self.encode(tuplets['pp_image'],tuplets['pp_trajectory'])
        z_pn = self.encode(tuplets['pn_image'],tuplets['pn_trajectory'])
        z_nn = self.encode(tuplets['nn_image'],tuplets['nn_trajectory'])
        anchor_pred = self.decode(z_a)
        
        loss_metric = self.metric_loss(z_a=z_a,
                                    z_pp=z_pp,
                                    z_pn=z_pn,
                                    z_nn=z_nn,
                                    d_in_pp=tuplets['pp_distance'],
                                    margin_nn=margin_nn,
                                    margin_pn=margin_pn,
                                    margin_pp=margin_pp,
                                    l2=l2,
                                    beta_pp=beta_pp,
                                    beta_pn=beta_pn,
                                    beta_nn=beta_nn,
                                    negatives_a = tuplets['negatives_anchor'],
                                    negatives_pp = tuplets['negatives_pp'],
                                    negatives_pn = tuplets['negatives_pn'],
                                    negatives_nn = tuplets['negatives_nn'],
                                    hardest_sampling = hardest_sampling)
        reconstruction_loss = self.reconstruction_loss_weighted(tuplets['anchor_target'],
                                                                anchor_pred,
                                                                beta_traj=beta_traj, 
                                                                beta_infra=beta_infra, 
                                                                beta_back_traj=beta_back_traj, 
                                                                beta_back_infra=beta_back_infra)
        loss = alpha_metric*loss_metric + alpha_recon*reconstruction_loss
        return loss