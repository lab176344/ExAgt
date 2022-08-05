from src.model.model import model 
from cv2 import transform
#from model import model 

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torchvision.models as models
from x_transformers import ContinuousTransformerWrapper, Encoder,TransformerWrapper
from torch.autograd import Variable
from src.model.model_101 import model_101 
#from model_101 import model_101 
import random
from einops import rearrange, reduce
encoderI_types = ["ViT","ResNet-18"]
encoderT_types = ["LSTM","Transformer-Encoder"]
merge_types =["Transformer"]
decoder_types = ["LSTM"]
import itertools

class traj_encoder(nn.Module):
    def __init__(self, *, dim_in, dim_out, depth, heads, dim_trans, dim_mlp):
        super().__init__()
        self.emb_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.model = ContinuousTransformerWrapper(
            dim_in = dim_in,
            max_seq_len = 51,
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
        mask = (torch.arange(seq_unpacked.shape[1])[None, :] < lens_unpacked[:, None]+1).to(seq_unpacked.device)
        x = self.model.forward(seq_unpacked,mask=mask)
        return self.mlp_head(x[:,0,:])


class merger_encoder(nn.Module):
    def __init__(self, *, dim_in=64, dim_trans=128, depth=6, heads=8):
        super().__init__()
        self.model = ContinuousTransformerWrapper(
            dim_in = dim_in,
            max_seq_len = 60,
            attn_layers = Encoder(
                dim = dim_trans,
                depth = depth,
                heads = heads
            )
        )
    def forward(self,x_infra,x_obj,obj_length,dim_i=64):
        x = torch.zeros((x_infra.shape[0],max(obj_length),dim_i)) # batch_size x valid_obj_length x emb_dim
        obj_length_padded = obj_length
        
        scenario_wise_merge = [] 
        # unpack objects and create scenario wise merge
        for i in range(x_infra.shape[0]):
            scenario_wise_merge_local = []
            scenario_wise_merge_local.append(x_infra[i].unsqueeze(0))
            scenario_wise_merge_local.append(x_obj[obj_length_padded[i]:obj_length_padded[i+1],:])
            scenario_wise_merge_local = [j for i in scenario_wise_merge_local for j in i]
            scenario_wise_merge_local = torch.stack(scenario_wise_merge_local,dim=0) # (infra + obj) x emb_dim
            scenario_wise_merge.append(scenario_wise_merge_local)
        scenario_wise_merge = torch.nn.utils.rnn.pad_sequence(scenario_wise_merge, batch_first=True) # batch_size x max(infra + obj) x emb_dim
        #scenario_wise_merge = scenario_wise_merge.to(x_infra.device) # write comment here why the hell I did 
        obj_length = obj_length+1 # added infra length
        mask = (torch.arange(scenario_wise_merge.shape[1])[None, :] < obj_length[1:, None]).to(x_infra.device) #,device = scenario_wise_merge.device # mask out irrelevant objects
        x = self.model.forward(scenario_wise_merge,mask=mask) # batch_size x max(infra + obj) x emb_dim
        return x, obj_length

class decoder_steer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            )
    def forward(self,x):
        x = self.model(x)
        return x

class lstm_decoder(nn.Module):    
    def __init__(self, input_size=None, hidden_size=None, output_size=None, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vor_linear = nn.Linear(hidden_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                            num_layers = num_layers, batch_first =True)
        self.nach_linear = nn.Linear(hidden_size, output_size)           

    def forward(self, x_input, encoder_hidden_states,t_steps):
        if t_steps==0:
            #encoder_hidden_states_0 = self.vor_linear(encoder_hidden_states[0])
            #encoder_hidden_states_0 = F.relu(encoder_hidden_states_0)
            #encoder_hidden_states_1 = encoder_hidden_states_0
            encoder_hidden_states = (encoder_hidden_states[0],encoder_hidden_states[0])
            output, self.hidden = self.lstm(x_input,encoder_hidden_states)
        else:
            output, self.hidden = self.lstm(x_input,encoder_hidden_states)
        output = self.nach_linear(output)          
        return output, self.hidden
    


class EncoderFull(nn.Module):
    def __init__(self, *, encoderI_type=encoderI_types[0], encoderT_type=encoderT_types[0], merge_type=merge_types[0],decoder_type=decoder_types[0], encoderI_args=None, encoderT_args=None,
                 z_dim_t=64,z_dim_i=64,zm_dim_in=64,zm_dim_out=128, m_depth=6, m_heads=8, image_size=200,channels=5,traj_size=3):
        super().__init__()
        assert encoderI_type in encoderI_types, 'EncoderI keyword unknown'
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert merge_type in merge_types, 'Merger keyword unknown'
        assert decoder_type in decoder_types, 'Decoder keyword unknown'
        self.encoderI_type = encoderI_type
        self.encoderT_type = encoderT_type
        self.merge_type = merge_type
        self.decoder_type = decoder_type

        self.dim_out_merge = zm_dim_out

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
            self.encoder_trajectory = nn.LSTM(input_size = traj_size, hidden_size = z_dim_t,batch_first =True)
        elif self.encoderT_type == encoderT_types[1]:
            #---------------------------------------
            #Transformer
            #---------------------------------------
            if not encoderT_args:
                self.encoder_trajectory = traj_encoder(dim_in = traj_size, dim_out=z_dim_t, depth=6, heads=8, dim_trans=64, dim_mlp=128)
            else:
                self.encoder_trajectory = traj_encoder(dim_in = traj_size, dim_out=z_dim_t, **encoderT_args)
        
        # Dynamic output limiting layer
        self.dynamic_limiter = nn.Tanh()

        # Linear transformation layer
        self.linear_transform = nn.Linear(2, 2)

        # Encoder Merger==========================================================================================================================
        if self.merge_type == merge_types[0]:
            self.merger_transformer = merger_encoder(dim_in= zm_dim_in, dim_trans=zm_dim_out, depth=m_depth, heads=m_heads)  
            
            
        # Decoder Steering
        if self.decoder_type == decoder_types[0]:
            self.deocder_lstm = lstm_decoder(input_size=traj_size, hidden_size=zm_dim_out, output_size=traj_size)
            
    def forward(self,
                x_image=None,
                x_traj=None,
                x_traj_len=None,
                batch_wise_object_lengths_sum=None,
                batch_wise_decoder_input=None,
                target_length=None,
                ):
        
        # Image Encoder==========================================================================================================================
        z_image = self.encoder_image.forward(x_image)
        
        # Trajectory Encoder==========================================================================================================================
        
        batch_wise_obj_pack_padded = torch.zeros((batch_wise_object_lengths_sum[-1],x_traj[0].shape[2],x_traj[0].shape[3]),device=x_image.device) # valid_objects x max_seq_len x feat_dim
        obj_length_padded = batch_wise_object_lengths_sum # valid objects sum over batch
        for unp in range(x_traj[0].shape[0]):
            batch_wise_obj_pack_padded[obj_length_padded[unp]:obj_length_padded[unp+1],:,:] = x_traj[0][unp,0:x_traj[1][unp],:,:]  # take only valid objects from padded objects
            
        #x_traj[0][torch.arange(x_traj[0].shape[0]),0:x_traj[1],:,:]       
                 
        batch_wise_obj_traj_padded = torch.nn.utils.rnn.pack_padded_sequence(batch_wise_obj_pack_padded, x_traj_len, batch_first=True, enforce_sorted=False) # traj should be pack padded, DO NOT TOUCH THE DATA HERE
        batch_wise_obj_traj_embedded = self.encoder_trajectory(batch_wise_obj_traj_padded) #
        if self.encoderT_type == encoderT_types[0]:
            z_obj = batch_wise_obj_traj_embedded[1][0][0,:,:] # LSTM: returns hidden state, cell state, take only hidden state valid_obj x embedding_dim
        elif  self.encoderT_type == encoderT_types[1]:
            z_obj = batch_wise_obj_traj_embedded
            
        # Encoder Merger==========================================================================================================================
        h_merged,obj_len_updated = self.merger_transformer(z_image,z_obj,batch_wise_object_lengths_sum)
        
        # Remove Infra from h_merged
        
        h_merged = h_merged[:,1:,:]

        # Decoder Dynamics==========================================================================================================================
        decoder_input = batch_wise_decoder_input # batch_size x max_obj_len x t[-1] (traj)
        decoder_hidden = h_merged
        #decoder_input_pack_padded = torch.nn.utils.rnn.pack_padded_sequence(decoder_input, x_traj[1], batch_first=True, enforce_sorted=False)
        decoder_hidden_pack_padded = torch.zeros((1,batch_wise_object_lengths_sum[-1],self.dim_out_merge),device=x_image.device)
        decoder_input_pack_padded = torch.zeros((batch_wise_object_lengths_sum[-1],1,3),device=x_image.device) # valid_objects x 1 x traj_dim

        decoder_input_pack_padded = torch.zeros((batch_wise_object_lengths_sum[-1],1,3),device=x_image.device) # valid_objects x 1 x traj_dim
        for unp in range(x_traj[0].shape[0]):
            decoder_hidden_pack_padded[:,obj_length_padded[unp]:obj_length_padded[unp+1],:] = decoder_hidden[unp,0:x_traj[1][unp],:]  
            decoder_input_pack_padded[obj_length_padded[unp]:obj_length_padded[unp+1],:,:] = decoder_input[unp,0:x_traj[1][unp],:].unsqueeze(1)           
        
        decoder_input = decoder_input_pack_padded.squeeze(1)
        decoder_hidden = (decoder_hidden_pack_padded,decoder_hidden_pack_padded)
        
        X = torch.zeros((decoder_input.shape[0],target_length),device=h_merged.device)
        Y = torch.zeros((decoder_input.shape[0],target_length),device=h_merged.device)
        T = torch.zeros((decoder_input.shape[0],target_length),device=h_merged.device)
        dynamic_1 = torch.zeros((decoder_input.shape[0],target_length),device=h_merged.device)
        dynamic_2 = torch.zeros((decoder_input.shape[0],target_length),device=h_merged.device)
        for t in range(target_length): 
            out = self.deocder_lstm(decoder_input_pack_padded,decoder_hidden,t)
            decoder_output, decoder_hidden = out[0],out[1] 
            #decoder_output = self.linear_transform(decoder_output)
            # decoder_input_pack_padded = decoder_output
            #decoder_output = self.dynamic_limiter(decoder_output)*0.1 # the limit should be on the output of the decoder
            decoder_output = decoder_output.squeeze(1)
            dynamic_1[:,t] = decoder_output[:,0]
            dynamic_2[:,t] = decoder_output[:,1]
            # Kinematic model

            if t==0:
                X[:,t] = decoder_input[:,0] + decoder_output[:,0] * (0.1*torch.ones_like(decoder_output[:,0],device=h_merged.device))
                Y[:,t] = decoder_input[:,1] + decoder_output[:,1] * (0.1*torch.ones_like(decoder_output[:,1],device=h_merged.device))
                T[:,t] = decoder_output[:,2]
            else:
                X[:,t] = X[:,t-1] + decoder_output[:,0] * (decoder_output[:,2]-T[:,t-1])#(0.1*torch.ones_like(decoder_output[:,0],device=h_merged.device)) # 
                Y[:,t] = Y[:,t-1] + decoder_output[:,1] * (decoder_output[:,2]-T[:,t-1])#(0.1*torch.ones_like(decoder_output[:,0],device=h_merged.device)) # 
                T[:,t] = decoder_output[:,2]
            decoder_input_pack_padded = torch.cat((X[:,t].unsqueeze(1),Y[:,t].unsqueeze(1),T[:,t].unsqueeze(1)),dim=1)
            decoder_input_pack_padded = decoder_input_pack_padded.unsqueeze(1)

        return {'z_image':z_image,'z_obj':z_obj,'h_merged':h_merged,'X':X,'Y':Y,'T':T,'dynamic_1':dynamic_1,'dynamic_2':dynamic_2}
        #return {'z_image':z_image,'z_obj':z_obj,'h_merged':h_merged,'X':d1,'Y':d2,'d1':d1,'d2':d2}

class model_120(model):
    def __init__(self, 
                encoderI_type=None, 
                encoderT_type=None, 
                merge_type=None, 
                decoder_type=None,
                encoderI_args=None,
                encoderT_args=None, 
                merge_args=None,
                z_dim_t=64,
                z_dim_i=64,
                zm_dim_in=64,
                zm_dim_out=128,
                image_size=200,
                channels=5,
                traj_size=3,
                m_depth=6,
                m_heads=8,
                idx = 120,
                name = 'ScenarioMerger',
                size = None,
                n_params = None,
                input_ = 'image_multichannel_vector',
                output = 'vector for objections predicted',
                task = 'Representaion Learning',
                description = 'Learning behaviour from latent space'
                ):
        
        super().__init__(idx,name,size,n_params,input_,output,task,description)
        
        self.image_size = image_size
        self.zm_dim_in = zm_dim_in
        self.zm_dim_out = zm_dim_out

        self.encoder = EncoderFull(encoderI_type=encoderI_type,
            encoderT_type=encoderT_type,
            merge_type=merge_type,
            decoder_type=decoder_type,
            encoderI_args=encoderI_args,
            encoderT_args=encoderT_args,
            z_dim_t=z_dim_t,
            z_dim_i=z_dim_i,
            zm_dim_in=self.zm_dim_in,
            zm_dim_out=self.zm_dim_out,
            m_heads= m_heads,
            m_depth=m_depth,
            image_size=self.image_size,
            channels=channels,
            traj_size=traj_size,
            )   

    def forward(self,
                x_image=None,
                x_traj=None,
                x_traj_len=None,
                batch_wise_object_lengths_sum=None,
                batch_wise_decoder_input=None,
                target_length=None):
        output_encode = self.encoder(x_image,
                                    x_traj,
                                    x_traj_len=x_traj_len,
                                    batch_wise_object_lengths_sum=batch_wise_object_lengths_sum,
                                    batch_wise_decoder_input=batch_wise_decoder_input,
                                    target_length=target_length,)
     
        return output_encode
    


def generate_model(**model_params):
    return model_120(**model_params)


if __name__ == '__main__':
    from torch.nn.parameter import Parameter
    device = torch.device('cuda')
    batches = 8

    # Image encoding
    image_examples = Variable(torch.randn(batches,5,120,120).to(device))
    # Resnet encoding
    resnetModel = models.resnet18()
    resnetModel.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnetModel.fc = nn.Linear(512,64, bias=True) 
    resnetModel = resnetModel.to(device)
    output1 = resnetModel(image_examples)
    x_infra = output1
    print(output1.shape)
    # ViT encoding
    image_examples = Variable(torch.randn(batches,5,120,120).to(device))
    ViT = model_101(image_size = 120,channels=5,z_dim = 64,patch_size = 20,dim = 256,depth = 10,heads = 16,mlp_dim = 128)
    output1 = resnetModel(image_examples)
    print(output1.shape)
    
    # Trajectory encoding
    # LSTM for one object
    
    
    batch_wise_objects_ego = []
    batch_wise_object_lengths = []
    batch_wise_decoder_in = []
    batch_wise_gTruthx = []
    batch_wise_gTruthy = []

    objects_ego_length = []
    decoder_in = []
    gTruth = []
    for _ in range(batches):
        objects_ego = []
        no_objects = random.randint(2,6)
        objects_ego = [Variable(torch.randn(random.randint(30,50),2).to(device)) if i>0 else Variable(torch.randn(50,2).to(device)) for i in range(no_objects) ]
        decoder_in = [Variable(torch.randn(1,2).to(device)) if i>0 else Variable(torch.randn(1,2).to(device)) for i in range(no_objects) ]
        gTruthx =  [Variable(torch.randn(10,1).to(device)) if i>0 else Variable(torch.randn(10,1).to(device)) for i in range(no_objects) ]
        gTruthy =  [Variable(torch.randn(10,1).to(device)) if i>0 else Variable(torch.randn(10,1).to(device)) for i in range(no_objects) ]

        objects_ego_length += [objects_ego[i].shape[0] for i in range(no_objects)]
        objects_ego = torch.nn.utils.rnn.pad_sequence(objects_ego, batch_first=True)
        decoder_in = torch.nn.utils.rnn.pad_sequence(decoder_in, batch_first=True)
        gTruthx = torch.nn.utils.rnn.pad_sequence(gTruthx, batch_first=True,padding_value=-100000)
        gTruthy = torch.nn.utils.rnn.pad_sequence(gTruthy, batch_first=True,padding_value=-100000)

        batch_wise_gTruthx.append(gTruthx)
        batch_wise_gTruthy.append(gTruthy)
        batch_wise_objects_ego.append(objects_ego)
        batch_wise_object_lengths.append(len(objects_ego))
        batch_wise_decoder_in.append(decoder_in)
    batch_wise_decoder_in = torch.nn.utils.rnn.pad_sequence(batch_wise_decoder_in, batch_first=True)
    batch_wise_decoder_in = batch_wise_decoder_in.squeeze(2)
    batch_wise_objects_ego_transformer = batch_wise_objects_ego
    batch_wise_gTruthx = torch.nn.utils.rnn.pad_sequence(batch_wise_gTruthx, batch_first=True,padding_value=-100000)
    batch_wise_gTruthy = torch.nn.utils.rnn.pad_sequence(batch_wise_gTruthy, batch_first=True,padding_value=-100000)

    batch_wise_objects_ego = torch.nn.utils.rnn.pad_sequence(batch_wise_objects_ego, batch_first=True)
    batch_wise_object_lengths_sum = torch.cumsum(torch.Tensor(batch_wise_object_lengths),0).int()
    objects_ego = objects_ego.to(device)
    
    obj_length_padded = batch_wise_object_lengths_sum

    encoder_trajectory = nn.LSTM(input_size = 2,hidden_size = 64, batch_first =True)
    encoder_trajectory = encoder_trajectory.to(device)
    
    packed_trajectory_m = torch.empty(batch_wise_object_lengths_sum[-1],batch_wise_objects_ego.shape[2],batch_wise_objects_ego.shape[3],requires_grad=True).cuda()

    for unp in range(batch_wise_objects_ego.shape[0]):
        packed_trajectory_m[obj_length_padded[unp]:obj_length_padded[unp+1],:,:] = batch_wise_objects_ego[unp,0:batch_wise_object_lengths[unp],:,:]
    
    traj_m = torch.nn.utils.rnn.pack_padded_sequence(packed_trajectory_m, objects_ego_length, batch_first=True, enforce_sorted=False)

    output_trajectory = encoder_trajectory(traj_m)
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output_trajectory[0],batch_first=True)

    print(output_trajectory[1][0][0,:,:].shape)
    x_obj = output_trajectory[1][0][0,:,:]
    # Transformer encoder
    transformer = traj_encoder(dim_in = 2, dim_out=64, depth=6, heads=8, dim_trans=64, dim_mlp=128)
    transformer = transformer.to(device)
    
    traj = torch.nn.utils.rnn.pack_padded_sequence(packed_trajectory_m, objects_ego_length, batch_first=True, enforce_sorted=False)
    output_trajectory = transformer(traj)
    #unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output_trajectory[0],batch_first=True)
    print(output_trajectory.shape)
    
    # Transformer merger
    merger_transformer = merger_encoder(dim_in=64, depth=6, heads=8,dim_trans=128)  
    merger_transformer = merger_transformer.to(device)
    output,obj_length = merger_transformer(x_infra,x_obj,batch_wise_object_lengths_sum)
    print(output.shape) 

    # remove infra before decoding
    output = output[:,1:,:]
    
    # MLP decoder
    neural_decoder = decoder_steer(128,20)
    neural_decoder = neural_decoder.to(device)

    steer_accel = neural_decoder(output)
    print(steer_accel.shape)
    
    # LSTM decoder
    lstm_decod = lstm_decoder(2,128,2)
    target_len = 10
    decoder_input = batch_wise_decoder_in
    decoder_hidden = output
    decoder_input_pack_padded = torch.nn.utils.rnn.pack_padded_sequence(decoder_input, batch_wise_object_lengths, batch_first=True, enforce_sorted=False)
    
    #decoder_hidden_pack_padded = torch.emo
    decoder_hidden_pack_padded = torch.empty(1,batch_wise_object_lengths_sum[-1],128,requires_grad=True).cuda()
    for unp in range(batch_wise_objects_ego.shape[0]):
        decoder_hidden_pack_padded[:,obj_length_padded[unp]:obj_length_padded[unp+1],:] = decoder_hidden[unp,0:batch_wise_object_lengths[unp],:]  
    
    deccoder_trajectory = nn.LSTM(input_size = 2, hidden_size = 128, batch_first =True)
    deccoder_trajectory = deccoder_trajectory.to(device)
    linear_trajectory_out = nn.Linear(128, 2)           
    linear_trajectory_out = linear_trajectory_out.to(device)
    decoder_input = decoder_input_pack_padded[0].unsqueeze(1)
    decoder_hidden = (decoder_hidden_pack_padded,decoder_hidden_pack_padded)
    X = torch.zeros(decoder_input.shape[0],target_len).to(device)
    Y = torch.zeros(decoder_input.shape[0],target_len).to(device)

    for t in range(target_len): 
        #decoder_hidden_pack_padded[0] = decoder_hidden_pack_padded[0].squeeze(1) 
        out = deccoder_trajectory(decoder_input,decoder_hidden)
        decoder_output, decoder_hidden = out[0],out[1] 
        output_model_param = linear_trajectory_out(decoder_output)
        print(output_model_param.shape)
        decoder_input = output_model_param
        output_model_param = output_model_param.squeeze(1)
        if t==0:
            X[:,t] = decoder_input_pack_padded[0][:,0] + output_model_param[:,0] * torch.ones_like(output_model_param[:,0])
            Y[:,t] = decoder_input_pack_padded[0][:,1] + output_model_param[:,1] * torch.ones_like(output_model_param[:,1])
        else:
            X[:,t] = X[:,t-1] + output_model_param[:,0] * torch.ones_like(output_model_param[:,0])
            Y[:,t] = Y[:,t-1] + output_model_param[:,1] * torch.ones_like(output_model_param[:,1])
    
    X_reshaped = torch.empty_like(batch_wise_gTruthx).to(batch_wise_gTruthx.device)
    Y_reshaped = torch.empty_like(batch_wise_gTruthx).to(batch_wise_gTruthx.device)

    for unp in range(batch_wise_objects_ego.shape[0]):
        X_reshaped[unp,0:batch_wise_object_lengths[unp],:,:] = X[obj_length_padded[unp]:obj_length_padded[unp+1],:].unsqueeze(2)  
        Y_reshaped[unp,0:batch_wise_object_lengths[unp],:,:] = Y[obj_length_padded[unp]:obj_length_padded[unp+1],:].unsqueeze(2)    

    maskX = batch_wise_gTruthx!=-100000
    maskY = batch_wise_gTruthy!=-100000

    X_mse = F.mse_loss(X_reshaped[maskX],batch_wise_gTruthx[maskX])
    Y_mse = F.mse_loss(Y_reshaped[maskY],batch_wise_gTruthy[maskY])
    loss = X_mse + Y_mse
    loss.backward()
    print(X.shape)
    print(Y.shape)
    
    
    # Losses
     
