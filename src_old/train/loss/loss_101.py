import torch
import torch.nn as nn
from src.train.loss.loss import loss
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

class loss_100(loss):
    def __init__(self,
                idx = 100,
                name = 'Double SWAV + Weighted Recon Loss',
                description = 'Double SWAV given: anchor, pp, pn +  Weighted Recon Loss',
                input_ = '',
                output = '1D',
                ) -> None:
        super().__init__(idx,name,description,input_,output)

    def swav_loss(self, output):
        #output (b,2,laten_dim)
        n_examples = output.shape[1]

        # ============ swav loss ... ============
        loss = 0
        for i in np.arange(n_examples):
            with torch.no_grad():
                out = output[:,i,:].detach()
                q = self.distributed_sinkhorn(out)

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(n_examples), i):
                x = output[:,v,:]
                subloss -= torch.mean(
                    torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (n_examples - 1)
        loss /= n_examples

        return loss

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

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
                data, 
                alpha_recon=1.0, 
                alpha_INFRAswav=1.0,
                alpha_MERGEswav=1.0, 
                beta_traj=1.0,
                beta_infra=1.0,
                beta_back_infra=1.0,
                beta_back_traj=1.0):
        """
        Computes loss for each batch.
        """

        batch_infra = torch.stack(data['z_infra_a'],data['z_infra_pp'],data['z_infra_pn'])
        batch_infra = torch.stack(data['z_merge_a'],data['z_merge_pp'])

        # SWAV
        loss_infra_swav = self.swav_loss(batch_infra)
        loss_merge_swav = self.swav_loss(batch_infra)


        # RECON
        reconstruction_loss = self.reconstruction_loss_weighted(data['image_target_a'],
                                                                data['image_out_a'],
                                                                beta_traj=beta_traj, 
                                                                beta_infra=beta_infra, 
                                                                beta_back_traj=beta_back_traj, 
                                                                beta_back_infra=beta_back_infra)
        reconstruction_loss += self.reconstruction_loss_weighted(data['image_target_pp'],
                                                                data['image_out_pp'],
                                                                beta_traj=beta_traj, 
                                                                beta_infra=beta_infra, 
                                                                beta_back_traj=beta_back_traj, 
                                                                beta_back_infra=beta_back_infra)
        reconstruction_loss += self.reconstruction_loss_weighted(data['image_target_pn'],
                                                                data['image_out_pn'],
                                                                beta_traj=beta_traj, 
                                                                beta_infra=beta_infra, 
                                                                beta_back_traj=beta_back_traj, 
                                                                beta_back_infra=beta_back_infra)
        reconstruction_loss /= 3.0
        loss = alpha_INFRAswav*loss_infra_swav + alpha_MERGEswav*loss_merge_swav  + alpha_recon*reconstruction_loss
        return loss