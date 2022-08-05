import torch
import torch.nn as nn
from src.model.model import model 
from torch.autograd import Variable
from scipy.linalg import eigh

#--------------------------- Graph Computations
def compute_laplacian(A):
    '''
    Compute Graph Laplacian Matrix given the Adjacency matrix.

    Parameters
    ----------
    A : [int, flaot] mat
        Adjancency matrix.

    Returns
    -------
    L : [int, float] mat
        Laplacian matrix.

    '''
    diag = np.sum(A, axis=0)
    D = np.zeros(A.shape)
    np.fill_diagonal(D, diag)
    L = D - A
    return L

def calc_2D_graph_fourier_torch(F, U_1, U_2):
    F_hat_mat = torch.matmul(U_1.transpose(-2, -1), torch.matmul(F, U_2))
    return F_hat_mat

def calc_inv_2D_graph_fourier_torch(F_hat, U_1, U_2):
    F = torch.matmul(U_1, torch.matmul(F_hat, U_2.transpose(-2,-1)))
    return F


def calc_graph_fourier_torch(f, U):
    F_hat = torch.matmul(f, U)
    return F_hat

def calc_inv_graph_fourier_torch(f_hat, U):
    F = torch.matmul(f_hat, U.transpose(0,1))
    return F
    

#--------------------------- Model Computations
def dist_2_weight(F_x, F_y):
    w = np.sqrt((2-F_x)**2 + (2-F_y)**2)
    return np.mean(w, axis=0)/3

def eucl_dist_2_weight(F_x, F_y):
    w = 1/(np.sqrt((F_x)**2 + (F_y)**2)+0.1)
    return np.mean(w, axis=0)


def rduce(rdc, EigVec0, s):
    n_rdc = int(EigVec0.shape[0]/rdc)
    new_EigVec0 = EigVec0[:, :n_rdc]
    new_s = s[:n_rdc, :]
    return new_s

#--------------------------- Models  
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim                        # Hidden dimensions
        self.layer_dim = layer_dim                          # Number of hidden layers
        # Building your LSTM -> input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)


    def forward(self, x):
        # Initialize hidden state with zeros
        # Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # One time step: output, (hidden_state, cell_state)'
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Index hidden state of last time step:   out = self.fc(out[:, -1, :]) 
        return out, (hn, cn)
      
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Gft2dBlock(nn.Module):
  def __init__(self, dim, rdc, bias=True):
    super().__init__()
    self.var = nn.Parameter(torch.rand(dim), requires_grad=True)
    self.rdc = rdc
  def forward(self, x, EigVec0, EigVec1):
    F_hat = calc_2D_graph_fourier_torch(x, torch.from_numpy(EigVec0).float(),
			                                torch.from_numpy(EigVec1).float())
    if self.rdc!= 1:
        F_hat = rduce(self.rdc, EigVec0, F_hat)			                           
    F_hat_flattened= F_hat.flatten().reshape(1, 1, -1)
    conv = self.var*F_hat_flattened
    n_rdc = int(EigVec0.shape[0]/self.rdc)
    F_re = calc_inv_2D_graph_fourier_torch(
             (conv).reshape(n_rdc, 9), torch.from_numpy(EigVec0[:, :n_rdc]).float(),
                                    torch.from_numpy(EigVec1).float())
    return conv.flatten(), F_hat, F_re


class gftnn2d(nn.Module):
    def __init__(self, nfeat, nhid, nout, nveh, depth, weighted, rdc, bias, dropout=0.):
        super().__init__()
        self.weighted = weighted
        self.nums = 2 if (weighted) else 4
        self.dim = int(nfeat*nveh/rdc)
        'Define Scenario Matrices'
        self.A_time = np.diagflat(np.ones(nfeat-1), 1) + np.diagflat(np.ones(nfeat-1), -1)
        self.EigVal0, self.EigVec0 = eigh(compute_laplacian(self.A_time)) # Eigendecomposition
        self.A_scene = np.zeros([nveh, nveh])
        if not self.weighted: # Laplacian is static when not weighted by distance of scenari
            self.A_scene[0,1:] = 1
            self.A_scene[1:,0] = 1
            self.EigVal1, self.EigVec1 = eigh(compute_laplacian(self.A_scene)) # Eigendecomposition
        else:
            self.EigVal1, self.EigVec1 = None, None
        self.gft2d_convs = nn.ModuleList([Gft2dBlock(self.dim, rdc, bias) for i in range(self.nums)])
        self.sep_norm = nn.ModuleList([
            PreNorm(self.dim, FeedForward(self.dim, nhid)) for i in range(self.nums)])
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(self.dim*self.nums, 
                        FeedForward(self.dim*self.nums, nhid, dropout = dropout))  ]))
        self.sigmoid = torch.nn.Sigmoid()
        self.func_out = torch.nn.Linear(self.dim*self.nums, nout)
        
    def forward(self, F_x, F_y, F_vx, F_vy):
        z = torch.tensor([])
        F_conv = torch.tensor([])
        if self.weighted:
            w = eucl_dist_2_weight(F_x.detach().numpy(), F_y.detach().numpy())
            self.A_scene[0 ,1:] = w[1:]
            self.A_scene[1:, 0]= w[1:]
            self.EigVal1, self.EigVec1 = eigh(compute_laplacian(self.A_scene))
            veh_feat = torch.stack((F_vx, F_vy), 0)
        else:
            veh_feat = torch.stack((F_x, F_y, F_vx, F_vy), 0)
        for i in range(self.nums):
            gft2d_conv, f_hat, f_re = self.gft2d_convs[i](veh_feat[i], self.EigVec0, self.EigVec1)
            z = torch.cat([z, self.sep_norm[i](gft2d_conv)], dim = 0)
            if F_conv.shape[0] == 0:
                F_conv = gft2d_conv.reshape(1,-1)
                F_hat = f_hat.reshape(1,-1)
                F_re = f_re.reshape(1,-1)
            else:
                F_conv = torch.cat([F_conv, gft2d_conv.reshape(1,-1)], dim=0)
                F_hat = torch.cat([F_hat, f_hat.reshape(1,-1)], dim=0)
                F_re = torch.cat([F_re, f_re.reshape(1,-1)], dim = 0)
        for ff in self.layers:
            z = ff[0](z)
        z = self.sigmoid(z)
        return self.func_out(z), F_conv, F_hat, F_re
        

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decode = SigmoidDecoder.apply
    def forward(self, z, v0):
        y_pred = self.decode(z, v0)
        return y_pred


class AI_Decoder(nn.Module):
    def __init__(self, latent_dim, layer_dim, output_dim=PRED_FRAMES):
        super().__init__()
        self.input_dim = latent_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(latent_dim, 16)
        self.fc2 = nn.Linear(16, 64)
        self.lstm_x = LSTMModel(64, output_dim, layer_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.lstm_y = LSTMModel(64, output_dim, layer_dim, output_dim)
        self.fc4 = nn.Linear(output_dim, output_dim)
        
    def forward(self, z):
        # decode path
        x = torch.tanh(self.fc1(z))
        x = torch.tanh(self.fc2(x))
        out_x, (hn0, cn0) = (self.lstm_x(x.reshape(1,1, -1)))
        dx = self.fc3(out_x) 
        out_y, (hn0, cn0) = (self.lstm_y(x.reshape(1,1, -1)))
        dy = self.fc4(out_y) 
        
        y_pred = torch.cat((dx.view(-1, dx.size(-1)), dy.view(-1, dy.size(-1))), dim=1)
        if torch.isnan(y_pred).any():
            raise ValueError('y has nan value', y_pred)
            y_pred = torch.nan_to_num(y_pred)
        return y_pred



class model_4(model):
    def __init__(self,
                 idx=4,
                 name="Marion Trajectory Prediction",
                 size=None,
                 n_params=None,
                 input=None,
                 output=None,
                 task=None,
                 description="Trajectory Prediction",
                 nfeat, nhid, nout, nveh, depth, weighted=False, rdc=1, bias=True, dropout=0, PRED_FRAMES=125) -> None:
        super().__init__(idx,name,size,n_params,input_,output,task,description)
        self.idx = idx
        self.name = name
        self.size = size
        self.n_params = n_params
        self.input = input
        self.output = output
        self.task = task
        self.description = description
        # --------------------------------------------- init encoder
        self.encoder = gftnn2d(nfeat, nhid, nout, nveh, depth, weighted, rdc, bias, dropout)
        # --------------------------------------------- init decoder
        #self.decoder = Decoder(nout, PRED_FRAMES)
        self.decoder = AI_Decoder(latent_dim=nout, layer_dim=1, output_dim=PRED_FRAMES)
        
    def forward(self, dataFx, dataFy, dataFvx, dataFvy, v0):
    # --------------------------------------------- encode
        z, F_conv, F_hat, F_re = self.encoder(dataFx, dataFy, dataFvx, dataFvy)
	# --------------------------------------------- decode
        y_hat = self.decoder(z)
	# --------------------------------------------- return
        return y_hat, z, F_conv, F_hat, F_re

