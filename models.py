# 
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse3t import *

# NN decoder
class Decoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3):
        super(Decoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.elu(h)
        outputs = self.output_linear(h)
        return outputs

# SNF, 3D space with time
class SNF3T(nn.Module):
    def __init__(self, prev_model = False):
        super(SNF3T, self).__init__()
        if prev_model is True:
            max_num_queries = 3 * N_batch
        else:
            max_num_queries = max(3 * N_batch, int(np.ceil((res_x+1)*(res_y+1)*(res_z+1)/num_chunks)))
        self.fp = Sparse3TEncoder(max_num_queries=max_num_queries)
        self.decoder1 = Decoder(D=1, W=64, input_ch=feat_dim * num_levels, output_ch=1)
        self.decoder2 = Decoder(D=1, W=64, input_ch=feat_dim * num_levels, output_ch=1)
        self.decoder3 = Decoder(D=1, W=64, input_ch=feat_dim * num_levels, output_ch=1)

    def eval_x(self, x_coord):
        interped = self.fp(x_coord)
        return self.decoder1(interped)
    
    def eval_y(self, y_coord):
        interped = self.fp(y_coord)
        return self.decoder2(interped)

    def eval_z(self, z_coord):
        interped = self.fp(z_coord)
        return self.decoder3(interped)

    def eval(self, coord):
        interped = self.fp(coord)
        decoded1 = self.decoder1(interped)
        decoded2 = self.decoder2(interped)
        decoded3 = self.decoder3(interped)
        return torch.cat((decoded1, decoded2, decoded3), dim = 1)

    def forward(self, coord):
        interped = self.fp(coord.view(-1, 4)).view(coord.shape[0], 3, -1)     
        decoded1 = self.decoder1(interped[..., [0], :])
        decoded2 = self.decoder2(interped[..., [1], :])
        decoded3 = self.decoder3(interped[..., [2], :])
        return torch.cat((decoded1, decoded2, decoded3), dim = 1)
