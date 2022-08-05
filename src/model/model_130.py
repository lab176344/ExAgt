from src.model.model import model 
import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder
from src.model.model_131 import model_131 as autoRegressiveWrapper

class Motion_Prediction(nn.Module):
    def __init__(self, dim_in, dim_out, depth, heads, dim_trans, dim_mlp,max_seq_len)-> None:
        super().__init__()
        self.encoder = ContinuousTransformerWrapper(
            dim_in = dim_in,
            max_seq_len = max_seq_len,
            attn_layers = Encoder(
                dim = dim_trans,
                depth = depth,
                heads = heads
            )
        )
        self.decoder = ContinuousTransformerWrapper(
            dim_in = dim_in,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                cross_attend = True,
                dim = dim_trans,
                depth = depth,
                heads = heads
            )
        )
        self.decoder = autoRegressiveWrapper(self.decoder)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_trans, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_out)
        )

    def forward(self,x,tgt):
        #seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        #mask = (torch.arange(x.shape[1])[None, :] < lens_unpacked[:, None]+1).to(seq_unpacked.get_device())
        encoder_out = self.encoder.forward(x)
        tgt = torch.cat((x[:,-1,:][:,None,:],tgt),dim=1)
        decoder_out = self.decoder.forward(tgt,context=encoder_out)
        outHead     = self.mlp_head(decoder_out)
        
        return outHead


class model_130(model):
    def __init__(self, 
                dim_in, dim_out, depth, heads, dim_trans, dim_mlp,max_seq_len,
                idx = 130,
                name = 'TODO',
                size = None,
                n_params = None,
                input_ = 'TODO',
                output = 'TODO',
                task = 'TODO',
                description = 'TODO'
                )->None:
        super().__init__(idx,name,size,n_params,input_,output,task,description)
        self.model = Motion_Prediction(dim_in, dim_out, depth, heads, dim_trans, dim_mlp,max_seq_len)
        self.model = self.model
    def forward(self,x,tgt):
        return self.model.forward(x,tgt)

def generate_model(**model_params):
    return model_130(**model_params)