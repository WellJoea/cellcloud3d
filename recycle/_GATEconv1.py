#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _GATE.py
* @Author  : Wei Zhou                                     *
* @Date    : 2023/09/15 21:47:12                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from tqdm import tqdm

from torch.nn.utils.parametrizations import weight_norm
# from torch.nn.utils import weight_norm

from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from cellcloud3d.alignment._gatconv import dGATConv, sGATConv
from cellcloud3d.alignment._utilis import seed_torch, shiftedReLU, Activation
from cellcloud3d.alignment._loss import loss_recon, loss_structure, loss_mse, loss_gae,loss_contrast,loss_soft_cosin

import numpy as np

class GATEConv(torch.nn.Module):
    '''
    seed_torch()
    hidden_dims = [data.x.shape[1]] + [512, 128, 48]
    model = GATEConv(hidden_dims,
                    Heads=1,
                    Concats=False,
                    bias=False,
                    gattype='gatv3')
    model(data.x, data.edge_index)
    '''
    def __init__(self,
                 hidden_dims,
                 Heads = 1,
                 Concats = False,
                 negative_slope = 0.2,
                 gattype = 'gatv2',
                 bias=False,
                 share_weights = False,
                 layer_attr = None,
                 tied_attr = None,
                 final_act= None,
                 hidden_act = None,
                 dropout=0,
                 **kargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.e_layers = len(hidden_dims) -1
        self.n_layers = self.e_layers * 2
        self.Heads = Heads
        self.Concats = Concats
        self.bias = bias
        self._layers()

        self.layer_attr = layer_attr
        self._tie_attr(tied_attr)

        self.dropout = dropout
        self.gattype = gattype or 'gatv3'
        self.loss_recon = loss_recon
        self.loss_structure = loss_structure
        self.loss_mse = loss_mse
        self.loss_gae = loss_gae
        self.loss_contrast = loss_contrast
        self.loss_soft_cosin = loss_soft_cosin

        self.final_act = Activation( final_act or 'linear', negative_slope=negative_slope)
        self.hidden_act = Activation( hidden_act or 'elu', negative_slope=negative_slope) 

        self.forwargs = {}
        self._autoedcoder(init_weights=True,
                      gattype = self.gattype,
                      bias = self.bias,
                      share_weights=share_weights,
                      negative_slope=negative_slope,
                      **kargs)

        # self.line_rese = nn.Linear(self.hidden_dims[0], 1 * self.hidden_dims[-1], bias=False)
        #self.line_rese = weight_norm(self.line_rese, name='weight', dim=0)

        # self.line_resd = nn.Linear(self.hidden_dims[-1], 1 * self.hidden_dims[0], bias=False)
        #self.line_resd = weight_norm(self.line_resd, name='weight', dim=0)

        print('model: ', self.autoedcoder)

    def _layers(self):
        if type(self.Heads) in [int]:
            self.Heads = [self.Heads] * self.n_layers
        elif type(self.Heads) in [list, np.ndarray]:
            assert len(self.Heads) == self.n_layers, 'the length of Heads must be same with layers.'
        else:
            raise('Error type of Heads.')

        if type(self.Concats) in [bool]:
            self.Concats = [self.Concats] * self.n_layers
        elif type(self.Concats) in [list, np.ndarray]:
            assert len(self.Concats) == self.n_layers, 'the length of Concats must be same with layers.'
            self.Concats = list(map(bool, self.Concats))
        else:
            raise('Error type of Concats.')

        self.Layers = [ [self.hidden_dims[i], self.hidden_dims[i+1]] for i in range(self.e_layers)]
        self.Layers += [ ly[::-1] for ly in self.Layers[::-1] ]

    def _tie_attr(self, tied_attr):
        self.tied_attr = {}
        for k, v in tied_attr.items():
            if not v is None:
                if v =='sym':
                    v = { self.n_layers-1-i : i for i in range(self.e_layers)}
                elif isinstance(v, list):
                    v = { self.n_layers-1-i : i for i in v}
                elif isinstance(v, dict):
                    v = v
                else:
                    raise('Error type of tied_attr.')
                for ik,iv in v.items():
                    assert (self.n_layers >ik >= self.e_layers) and (iv <self.e_layers)
                self.tied_attr[k] = v

    def _autoedcoder(self, **kargs):
        self.autoedcoder = nn.ModuleList()
        for layer in range(self.n_layers):
            iargs = {**kargs,
                    "heads" : self.Heads[layer],
                    "concat": self.Concats[layer] }
            if self.layer_attr:
                iargs.update(self.layer_attr.get(layer,{}))
            iconv = self.GConv(*self.Layers[layer])
            try:
                iconv = self.GConv(*self.Layers[layer], **iargs)
            except:
                for k, v in iargs.items():
                    if not hasattr(iconv, k):
                        print(f'delete arg {k} -> {v}')
                        del iargs[k]
                iconv = self.GConv(*self.Layers[layer], **iargs)
            self.autoedcoder.append(iconv)

    def _tie_attr(self, tied_attr):
        self.tied_attr = {}

        for k, v in tied_attr.items():
            if not v is None:
                if v =='sym':
                    v = { self.n_layers-1-i : i for i in range(self.e_layers)}
                elif isinstance(v, list):
                    v = { self.n_layers-1-i : i for i in v}
                elif isinstance(v, dict):
                    v = v
                else:
                    raise('Error type of tied_attr.')
                self.tied_attr[k] = v
        print(self.tied_attr)

    def _tie_weight(self, dConv, dlayer):
        karg = {}
        for k, v in self.tied_attr.items():
            if k in ['W'] and v.get(dlayer):
                eConv = self.autoedcoder[v.get(dlayer)]
                try:
                    dConv.lin_l.weight = eConv.lin_l.weight.transpose(0, 1)
                    dConv.lin_r.weight = eConv.lin_r.weight.transpose(0, 1)
                except:
                    try:
                        dConv.lin_l.weight = nn.Parameter(eConv.lin_l.weight.transpose(0, 1))
                        dConv.lin_r.weight = nn.Parameter(eConv.lin_r.weight.transpose(0, 1))
                    except:
                        dConv.lin_l.data = nn.Parameter(eConv.lin_l.data.transpose(0, 1))
                        dConv.lin_r.data = nn.Parameter(eConv.lin_r.data.transpose(0, 1))
            if k in ['R'] and v.get(dlayer):
                eConv = self.autoedcoder[v.get(dlayer)]
                if not eConv.res_fc is None:
                    try:
                        dConv.res_fc.weight = eConv.res_fc.weight.transpose(0, 1)
                    except:
                        try:
                            dConv.res_fc.weight = nn.Parameter(eConv.res_fc.weight.transpose(0, 1))
                        except:
                            pass
            if k in ['A'] and v.get(dlayer):
                eConv = self.autoedcoder[v.get(dlayer)]
                karg['tie_alpha'] = eConv.tie_alpha

        return dConv, karg

    def forward(self,
                X: Union[Tensor, PairTensor],
                edge_index: Adj,
                edge_attr: OptTensor = None,
                ):
        # Encoder
        H = X

        # L = self.line_rese(X)
        for elayer in range(self.e_layers):
            eConv = self.autoedcoder[elayer]
            H = eConv(H, edge_index, 
                        edge_attr=edge_attr, 
                        **self.forwargs
                        )
            if elayer != (self.e_layers - 1):
                H = self.hidden_act(H)
                # H = F.dropout(H, p=self.dropout, training=self.training)

        # Representation
        #self.latent = H.view(-1, self.Heads[-1], self.Layers[-1][1]).mean(dim=1)
        # H =  H + L 
        self.H = H

        try:
            self.attention = eConv.alpha
        except:
            self.attention = None

        # Decoder
        # self.line_resd.weight = self.line_rese.weight.transpose(0, 1)
        # M = self.line_resd(H)
        for dlayer in range(self.e_layers, self.n_layers):
            dConv = self.autoedcoder[dlayer]
            dConv, karg = self._tie_weight(dConv, dlayer)
            H = dConv(H, edge_index, 
                        edge_attr=edge_attr,
                        **karg,
                        **self.forwargs
                        )

            if dlayer != (self.n_layers - 1):
                H = self.hidden_act(H)
                # H = F.dropout(H, p=self.dropout, training=self.training)
            else:
                H = self.final_act(H)

        self.X_ = H
        return self.H, self.X_, self.attention