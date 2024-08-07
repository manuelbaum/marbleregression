import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import math

def positional_encoding(seq_len, d_model):
    """
    Generates a positional encoding matrix for a sequence.

    Args:
        seq_len (int): The length of the sequence.
        d_model (int): The embedding dimension.

    Returns:
        torch.Tensor: A tensor of shape (seq_len, d_model) containing the positional encodings.
    """
    pe = torch.zeros(seq_len, d_model, device="cuda")
    position = torch.arange(0, seq_len, dtype=torch.float, device="cuda").unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device="cuda").float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

class AddPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(AddPositionalEncoding, self).__init__()
        self.dim = dim
        #self.n_batch = n_batch

    def forward(self, x):
        #x = x.view(self.n_batch, -1, self.dim)
        n_timesteps = x.size()[1]
        encoding = positional_encoding(n_timesteps, self.dim)
        x = x+encoding
        #x = x.view(self.n_batch*n_timesteps, self.dim)
        return x

class AverageDimension(nn.Module):
    def __init__(self, dim):
        super(AverageDimension, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=False)

class CastTorch(nn.Module):
    def __init__(self, type):
        super(CastTorch, self).__init__()
        self.type = type

    def forward(self, x):
        return x.to(self.type)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
class MLPflat(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_batch):
        super().__init__()

        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        af = nn.Sigmoid
        self.mlp = nn.Sequential(OrderedDict([

            #('average', AverageDimension(1)),# average out time
            #('norm1', nn.LayerNorm([self.input_dims[i_head]])),

            ('linear1', nn.Linear(input_dim, hidden_dim)),  # hidden_dims[i_head])),#
            ('relu1', af()),
            ('dropout', self.dropout),
            #('positional_encoding', AddPositionalEncoding(hidden_dim)),
            ('linear2', nn.Linear(hidden_dim, hidden_dim)),
            ('relu2', af()),
            ('dropout', self.dropout),
            # ('linear3', nn.Linear(hidden_dim, hidden_dim)),
            # ('relu3', nn.ReLU()),
            # ('linear4', nn.Linear(hidden_dim, output_dim)),

            # ('norm2', nn.LayerNorm([self.aggregate_dim])),
            # ('linear2', nn.Linear(hidden_dims[i_head], self.aggregate_dim)),
            # ('relu2', nn.ReLU())
        ]))

        self.mlp_out = nn.Sequential(OrderedDict([

            #('average', AverageDimension(1)),# average out time
            # ('norm1', nn.LayerNorm([self.input_dims[i_head]])),
            #('linear1', nn.Linear(hidden_dim, hidden_dim)),  # hidden_dims[i_head])),#

            # ('relu1', nn.ReLU()),
            # ('linear2', nn.Linear(hidden_dim, output_dim)),
            # ('relu2', nn.ReLU()),
            # ('linear3', nn.Linear(hidden_dim, hidden_dim)),
            ('dropout', self.dropout),
            ('relu3', af()),
            ('linear4', nn.Linear(hidden_dim, output_dim)),

            # ('norm2', nn.LayerNorm([self.aggregate_dim])),
            # ('linear2', nn.Linear(hidden_dims[i_head], self.aggregate_dim)),
            # ('relu2', nn.ReLU())
        ]))

    def latent(self, x):
        n_batch, n_timesteps = x.size()[0], x.size()[1]
        x = x.reshape(n_batch*n_timesteps,-1)
        x = self.batchnorm(x)
        x = x.reshape(n_batch, n_timesteps, -1)
        x = self.mlp(x)

        x = AverageDimension(1)(x)
        return x
    def forward(self, x):

        x = self.latent(x)

        x = self.batchnorm2(x)
        x = self.mlp_out(x)
        return x

class CNNMLPflat(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, cnn_flat_dim, hidden_dim_cnn_mlp, output_dim):
        super().__init__()

        n_channels = 8
        self.cnn_flat_dim = 32#int(cnn_flat_dim/16*n_channels)
        self.hidden_dim_cnn_mlp = hidden_dim_cnn_mlp
        self.dropout = nn.Dropout(p=0.1)
        af = nn.Sigmoid
        dtype = torch.float32
        self.mlp = nn.Sequential(OrderedDict([
            ('batchnorm2d', nn.BatchNorm2d(3)),
            ('conv1', nn.Conv2d(3, n_channels, 3, padding=1, stride=2, dtype=dtype)),  # First convolutional layer
            ('pool1', nn.MaxPool2d(2, 2)),  # Max pooling layer
            ('conv2', nn.Conv2d(n_channels, n_channels, 3, padding=1, stride=2, dtype=dtype)),  # Second convolutional layer
            ('pool2', nn.MaxPool2d(2, 2)),  # Max pooling layer
            ('conv3', nn.Conv2d(n_channels, n_channels, 3, padding=1, stride=2, dtype=dtype)),
            ('pool3', nn.MaxPool2d(2, 2)),  # Max pooling layer
            ('flatten', Flatten()),
            ('linear1', nn.Linear(self.cnn_flat_dim, self.hidden_dim_cnn_mlp)),  # hidden_dims[i_head])),#
            ('relu1', af()),
            ('dropout', self.dropout)

        ]))

        self.linear_preaggregate = nn.Linear(hidden_dim_cnn_mlp, hidden_dim_cnn_mlp)
        self.mlp_out = nn.Sequential(OrderedDict([

            # ('average', AverageDimension(1)),# average out time
            # ('norm1', nn.LayerNorm([self.input_dims[i_head]])),
            ('linear1', nn.Linear(self.hidden_dim_cnn_mlp, self.hidden_dim_cnn_mlp)),  # hidden_dims[i_head])),#
            ('dropout', self.dropout),
            ('relu1', af()),
            ('linear2', nn.Linear(self.hidden_dim_cnn_mlp, output_dim)),
            # ('relu2', nn.ReLU()),
            # ('linear3', nn.Linear(hidden_dim, hidden_dim)),
            # ('relu3', nn.ReLU()),
            # ('linear4', nn.Linear(hidden_dim, output_dim)),

            # ('norm2', nn.LayerNorm([self.aggregate_dim])),
            # ('linear2', nn.Linear(hidden_dims[i_head], self.aggregate_dim)),
            # ('relu2', nn.ReLU())
        ]))


    def latent(self, x):
        # x = AverageDimension(1)(x)
        # print("---")
        # x = x.to(torch.float16)
        # print(x.dtype)
        n_batch, n_timesteps = x.size()[0], x.size()[1]
        im_h, im_w = 120, 160
        x = x.reshape(n_batch, -1, im_h, im_w, 3)
        # plt.imshow(x[0,0].cpu().int())
        # plt.show()
        x = torch.permute(x, (0, 1, 4, 2, 3))  # rotate channels to the front

        # print(x.size())
        # xs = torch.zeros(n_batch,0,self.hidden_dim_cnn_mlp).cuda()
        x = torch.flatten(x, 0, 1)
        # for t in range(x.size()[1]):
        #     #print(x[:,t,:])
        #     tmp = self.mlp(x[:,t,:]).unsqueeze(1)
        #     # print(tmp)
        #     xs = torch.concat((xs,tmp),dim=1)
        # print(x.dtype)
        x = self.mlp(x)  # .unsqueeze(1)
        # print(x.dtype)
        # print('nan in x',torch.sum(torch.isnan(x)))
        x = x.reshape(n_batch, n_timesteps, -1)

        # pos_enc = positional_encoding(n_timesteps, self.hidden_dim_cnn_mlp)
        # x = x + pos_enc
        x = self.linear_preaggregate(x)
        x = F.sigmoid(x)
        x = self.dropout(x)

        x = torch.mean(x, dim=1)

        return x
    def forward(self, x):
        x = self.latent(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        #print('nan in x_mean', torch.sum(torch.isnan(x)))
        #print(x)
        out = self.mlp_out(x)
        #print('nan in x_out', torch.sum(torch.isnan(out)))
        #out = xs

        return out