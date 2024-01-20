import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv


from cellcloud3d.alignment._gatconv import edGATConv

from cellcloud3d.alignment._GATEconv import GATEConv as STAGATE

class STAGATE2(torch.nn.Module):
    def __init__(self, hidden_dims, **kargs):
        super(STAGATE, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = edGATConv(in_dim, num_hidden, heads=1, concat=False, init_gain=None,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = edGATConv(num_hidden, out_dim, heads=1, concat=False, init_gain=None,
                             dropout=0, add_self_loops=False, bias=False, aslinear=True)
        self.conv3 = edGATConv(out_dim, num_hidden, heads=1, concat=False, init_gain=None,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = edGATConv(num_hidden, in_dim, heads=1, concat=False, init_gain=None,
                             dropout=0, add_self_loops=False, bias=False, aslinear=True)
        print(self.conv4)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index)
        # try:
        #     self.conv3.lin_s.weight = self.conv2.lin_s.weight.transpose(0, 1)
        #     self.conv3.lin_d.weight = self.conv2.lin_d.weight.transpose(0, 1)
        #     self.conv4.lin_s.weight = self.conv1.lin_s.weight.transpose(0, 1)
        #     self.conv4.lin_d.weight = self.conv1.lin_d.weight.transpose(0, 1)
        # except:
        self.conv3.lin_s.weight = nn.Parameter(self.conv2.lin_s.weight.transpose(0, 1))
        self.conv3.lin_d.weight = nn.Parameter(self.conv2.lin_d.weight.transpose(0, 1))
        self.conv4.lin_s.weight = nn.Parameter(self.conv1.lin_s.weight.transpose(0, 1))
        self.conv4.lin_d.weight = nn.Parameter(self.conv1.lin_d.weight.transpose(0, 1))

        h3 = F.elu(self.conv3(h2, edge_index, tie_alpha=self.conv1.tie_alpha))
        h4 = self.conv4(h3, edge_index)

        return h2, h4  # F.log_softmax(x, dim=-1)
    
class STAGATE1(torch.nn.Module):
    def __init__(self, hidden_dims, **kargs):
        super(STAGATE, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)