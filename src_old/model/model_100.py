from src.model.model import model 
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torchvision.models as models
from x_transformers import ContinuousTransformerWrapper, Encoder

from src.model.model_101 import model_101 

encoderI_types = ["ViT","ResNet-18"]
encoderT_types = ["LSTM","Transformer-Encoder"]
merge_types =["FC"]

class traj_encoder(nn.Module):
    def __init__(self, *, dim_in, dim_out, depth, heads, dim_trans, dim_mlp):
        super().__init__()
        self.emb_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.model = ContinuousTransformerWrapper(
            dim_in = dim_in,
            max_seq_len = 31,
            attn_layers = Encoder(
                dim = dim_trans,
                depth = depth,
                heads = heads
            )
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_trans, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_out)
        )
    def forward(self,x):
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        emb_tokens = self.emb_token.expand(seq_unpacked.shape[0], -1, -1)
        seq_unpacked = torch.cat((emb_tokens, seq_unpacked), dim=1)
        mask = (torch.arange(seq_unpacked.shape[1])[None, :] < lens_unpacked[:, None]+1).to(seq_unpacked.get_device())
        x = self.model.forward(seq_unpacked,mask=mask)
        return self.mlp_head(x[:,0,:])

class FC(nn.Module):
    def __init__(self,n_layers,dims):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module("Lin_"+str(i),nn.Linear(dims[i][0],dims[i][1]))
            if i== n_layers-1:
                break
            self.layers.add_module("Activation_"+str(i),nn.ReLU())
    def forward(self, z_image,z_traj):
        x = torch.cat((z_image,z_traj),1)
        x = self.layers(x)
        return x

class EncoderFull(nn.Module):
    def __init__(self, *, encoderI_type=encoderI_types[0], encoderT_type=encoderT_types[0], merge_type=merge_types[0], encoderI_args=None, encoderT_args=None, merge_args=None,z_dim_t=32,z_dim_i=64,z_dim_m=64,image_size=200,channels=1,traj_size=3,nmb_prototypes_infra=512,nmb_prototypes_merge=512):
        super().__init__()
        assert encoderI_type in encoderI_types, 'EncoderI keyword unknown'
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert merge_type in merge_types, 'Merger keyword unknown'
        self.encoderI_type = encoderI_type
        self.encoderT_type = encoderT_type
        self.merge_type = merge_type

        self.infra_prototypes = nn.Linear(z_dim_i, nmb_prototypes_infra, bias=False)
        self.merge_prototypes = nn.Linear(z_dim_m, nmb_prototypes_merge, bias=False)

        # Image Encoder==========================================================================================================================
        if self.encoderI_type == encoderI_types[0]:
            #---------------------------------------
            #ViT
            #---------------------------------------
            if not encoderI_args:
                self.encoder_image = model_101(image_size = image_size,channels=channels,z_dim = z_dim_i,patch_size = 20,dim = 256,depth = 10,heads = 16,mlp_dim = 128)
            else:
                self.encoder_image = model_101(image_size = image_size,channels=channels,z_dim = z_dim_i,**encoderI_args)
        elif self.encoderI_type == encoderI_types[1]:
            #---------------------------------------
            #ResNet-18
            #---------------------------------------
            self.encoder_image  = models.resnet18()
            self.encoder_image.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder_image.fc = nn.Linear(512,z_dim_i, bias=True)
        
        # Trajectory Encoder==========================================================================================================================
        if self.encoderT_type == encoderT_types[0]:
            #---------------------------------------
            #LSTM
            #---------------------------------------
            self.encoder_trajectory = nn.LSTM(input_size = traj_size,hidden_size = z_dim_t,batch_first =True)
        elif self.encoderT_type == encoderT_types[1]:
            #---------------------------------------
            #Transformer
            #---------------------------------------
            if not encoderT_args:
                self.encoder_trajectory = traj_encoder(dim_in = traj_size, dim_out=z_dim_t, depth=6, heads=8, dim_trans=64, dim_mlp=128)
            else:
                self.encoder_trajectory = traj_encoder(dim_in = traj_size, dim_out=z_dim_t, **encoderT_args)
        

        # Encoder Merger==========================================================================================================================
        if self.merge_type == merge_types[0]:
            #---------------------------------------
            # FC      
            #---------------------------------------
            dims = [[z_dim_t+z_dim_i,z_dim_t+z_dim_i]]
            if not merge_args:
                n_layers = 2
                for i in range(n_layers-2):
                    dims.append([z_dim_t+z_dim_i,z_dim_t+z_dim_i])
                dims.append([z_dim_t+z_dim_i,z_dim_m])
                self.encoder_merge = FC(n_layers=n_layers,dims=dims)
            else:
                n_layers = merge_args['n_layers']
                for i in range(n_layers-2):
                    dims.append([z_dim_t+z_dim_i,z_dim_t+z_dim_i])
                dims.append([z_dim_t+z_dim_i,z_dim_m])
                self.encoder_merge = FC(dims=dims,**merge_args)

    def forward(self,
                x_image,
                x_traj,
                return_infra_z=False, return_infra_cXz=False, return_infra_q=False,
                return_merge_cXz=False, return_merge_q=False):
        z_image = self.encoder_image.forward(x_image)
        first_element = ((x_traj[:,1:,2]==0)==0).sum(dim=1)+1 
        packed_trajectory = torch.nn.utils.rnn.pack_padded_sequence(x_traj, first_element.cpu().numpy(), batch_first=True, enforce_sorted=False)
        z_traj = self.encoder_trajectory.forward(packed_trajectory)
        if self.encoderT_type == encoderT_types[0]:
            z_traj = z_traj[1][0][0,:,:]        
        z_merged = self.encoder_merge.forward(z_image,z_traj)
        
        Output ={'z':z_merged}
        if return_infra_z:
            Output['z_infra'] = z_image
        if return_infra_cXz:
            #TODO
            Output['cXz_infra'] = self.infra_prototypes(z_image)
        if return_infra_q:
            #TODO
            Output['q_infra'] = z_image
        if return_merge_cXz:
            #TODO
            Output['cXz_merge'] = self.infra_prototypes(z_merged)
        if return_merge_q:
            #TODO
            Output['q_merge'] = z_image
        
        return Output

    def normalize_prototypes(self):
        w = self.infra_prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.infra_prototypes.weight.copy_(w)
        w = self.merge_prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.merge_prototypes.weight.copy_(w)
        return True

class model_100(model):
    def __init__(self, 
                encoderI_type=None, 
                encoderT_type=None, 
                merge_type=None, 
                encoderI_args=None,
                encoderT_args=None, 
                merge_args=None,
                z_dim_t=32,
                z_dim_i=64,
                z_dim_m=64,
                image_size=200,
                channels=1,
                traj_size=3,

                idx = 100,
                name = 'ScenarioMergerDoubleSWAV',
                size = None,
                n_params = None,
                input_ = 'image_vector',
                output = 'image_vector_merged',
                task = 'Representaion Learning',
                description = 'Learning representaions and prototypes for the merged latent space as for the infrastructure laten space'
                ):
        
        super().__init__(idx,name,size,n_params,input_,output,task,description)
        
        self.image_size = image_size
        self.z_dim_m = z_dim_m
        self.encoder = EncoderFull(encoderI_type=encoderI_type,
            encoderT_type=encoderT_type,
            merge_type=merge_type,
            encoderI_args=encoderI_args,
            encoderT_args=encoderT_args,
            merge_args=merge_args,
            z_dim_t=z_dim_t,
            z_dim_i=z_dim_i,
            z_dim_m=self.z_dim_m,
            image_size=self.image_size,
            channels=channels,
            traj_size=traj_size,
            )
        
        if self.image_size == 400:
            self.layerDec1 = nn.ConvTranspose2d(self.z_dim_m, 32,5,2)
            self.layerDec2 = nn.ConvTranspose2d(32, 64,7,4)
            self.layerDec3 = nn.ConvTranspose2d(64, 64,10,4)
            self.layerDec4 = nn.ConvTranspose2d(64, 2,12,4)
        elif self.image_size == 200:
            self.layerDec1 = nn.ConvTranspose2d(self.z_dim_m, 32,6,2)
            self.layerDec2 = nn.ConvTranspose2d(32, 64,10,2)
            self.layerDec3 = nn.ConvTranspose2d(64, 64,10,2)
            self.layerDec4 = nn.ConvTranspose2d(64, 2,12,4)
        elif self.image_size == 100:
            self.layerDec1 = nn.ConvTranspose2d(self.z_dim_m, 64,6,2)
            self.layerDec2 = nn.ConvTranspose2d(64, 128,8,2)
            self.layerDec3 = nn.ConvTranspose2d(128, 128,8,2)
            self.layerDec4 = nn.ConvTranspose2d(128, 128,8,2)
            self.layerDec5 = nn.ConvTranspose2d(128, 64,6,1)
            self.layerDec6 = nn.ConvTranspose2d(64, 2,6,1)
        

    def encode(self, 
                x_image,
                x_traj,
                return_infra_z=False, return_infra_cXz=False, return_infra_q=False,
                return_merge_cXz=False, return_merge_q=False):
        Output = self.encoder.forward(x_image,
                                    x_traj,
                                    return_infra_z=return_infra_z, return_infra_cXz=return_infra_cXz, return_infra_q=return_infra_q,
                                    return_merge_cXz=return_merge_cXz, return_merge_q=return_merge_q)
        return Output
        
    
    def decode(self, z):
        if self.image_size == 100:
            x = z.view((z.size(0),z.size(1),1,1))
            x = self.layerDec1(x)
            x = F.relu(x)
            x = self.layerDec2(x)
            x = F.relu(x)
            x = self.layerDec3(x)
            x = F.relu(x)
            x = self.layerDec4(x)
            x = F.relu(x)
            x = self.layerDec5(x)
            x = F.relu(x)
            x = self.layerDec6(x)
            x = torch.sigmoid(x)
        else:
            print('Decode path not implemented')
            quit()
        return x

    def forward(self,
                x_image,
                x_traj,
                return_infra_z=False, return_infra_cXz=False, return_infra_q=False,
                return_merge_cXz=False, return_merge_q=False):
        output_encode = self.encode(x_image,
                                    x_traj,
                                    return_infra_z=return_infra_z, return_infra_cXz=return_infra_cXz, return_infra_q=return_infra_q,
                                    return_merge_cXz=return_merge_cXz, return_merge_q=return_merge_q)
        x = self.decode(output_encode['z'])
        output = output_encode
        output['x'] = x
        return output
    
    def normalize_prototypes(self):
        self.encoder.normalize_prototypes()
        return True

def generate_model(**model_params):
    return model_100(**model_params)