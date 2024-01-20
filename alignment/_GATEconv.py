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
from cellcloud3d.alignment._gatconv import dGATConv, edGATConv, sGATConv
from cellcloud3d.alignment._glconv import kGNN
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
                    )
    model(data.x, data.edge_index)
    '''
    def __init__(self,
                 hidden_dims,
                 Heads = 1,
                 Concats = False,
                 negative_slope = 0.2,
                 gconvs = ['gatv3','gatv3'],
                 share_weights = True,
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
        self._layers()
        self._gconvs(gconvs)
        self.layer_attr = layer_attr
        self._tie_attr(tied_attr or {})

        self.dropout = dropout

        self.loss_recon = loss_recon
        self.loss_structure = loss_structure
        self.loss_mse = loss_mse
        self.loss_gae = loss_gae
        self.loss_contrast = loss_contrast
        self.loss_soft_cosin = loss_soft_cosin

        self.final_act = Activation( final_act or 'linear', negative_slope=negative_slope)
        self.hidden_act = Activation( hidden_act or 'elu', negative_slope=negative_slope) 

        self._autoedcoder(init_weights=True,
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

    def _gconvs(self, gconvs):
        self.gconvs = [gconvs]*self.e_layers if isinstance(gconvs, str) else gconvs
        if len(self.gconvs) == self.e_layers:
            self.gconvs = self.gconvs + self.gconvs[::-1]
        if len(self.gconvs) != self.n_layers:
            raise('The length of gconvs must be same with layers.')

        self.GCONVs = []
        self.forwargs = []
        for igconv in self.gconvs:
            if igconv in ['gatv1']:
                gc = sGATConv
                ga = {'edge_attr': None, 'edge_weight': None}
            elif igconv in ['gatv2']:
                gc = dGATConv
                ga = {'edge_attr': None, 'edge_weight': None}
            elif igconv in ['gatv3']:
                gc = edGATConv
                ga = {'edge_attr': None, 'edge_weight': None}
            elif igconv in ['gatr1']:
                from torch_geometric.nn import GATConv
                gc = GATConv
                ga = {'return_attention_weights': None}
            elif igconv in ['gatr2']:
                from torch_geometric.nn import GATv2Conv
                gc = GATv2Conv
                ga = {'return_attention_weights': None}
            elif igconv in ['sage']:
                from torch_geometric.nn import SAGEConv
                gc = SAGEConv
                ga = {}
            elif igconv in ['tf']:
                from torch_geometric.nn import TransformerConv
                gc = TransformerConv
                ga = {}
            else:
                gc = gconvs
                ga = {}
            self.GCONVs.append(gc)
            self.forwargs.append(ga)

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

    def _init_layers(self, GConv, layer, **kargs):
        iargs = {**kargs,
                "heads" : self.Heads[layer],
                "concat": self.Concats[layer] }
        if self.layer_attr:
            iargs.update(self.layer_attr.get(layer,{}))
        iconv = GConv(*self.Layers[layer])
        try:
            iconv = GConv(*self.Layers[layer], **iargs)
        except:
            keys = list(iargs.keys())
            for k in keys:
                v = iargs[k]
                if not hasattr(iconv, k):
                    print(f'delete arg {k} -> {v}')
                    del iargs[k]
            iconv = GConv(*self.Layers[layer], **iargs)
        return iconv

    def _autoedcoder1(self, **kargs):
        self.autoedcoder = nn.ModuleList()
        from torch_geometric.nn.conv import GCNConv, FiLMConv, GraphConv,GatedGraphConv,ResGatedGraphConv,ARMAConv
        
        from torch_geometric.nn.conv import FusedGATConv, TAGConv, SGConv, RGCNConv, GENConv, FAConv, LGConv
        from torch_geometric.nn.conv import RGATConv, DNAConv, EGConv
        from torch_geometric.nn import SAGEConv, TransformerConv
        conv0 = self._init_layers(edGATConv, 0, **kargs)
        conv1 = self._init_layers(edGATConv, 1, **kargs)
        conv2 = self._init_layers(edGATConv, 2, **kargs)
        conv3 = self._init_layers(edGATConv, 3, **kargs)
        self.autoedcoder.append(conv0)
        self.autoedcoder.append(conv1)
        self.autoedcoder.append(conv2)
        self.autoedcoder.append(conv3)

    def _autoedcoder(self, **kargs):
        self.autoedcoder = nn.ModuleList()
        for layer in range(self.n_layers):
            iconv = self._init_layers(self.GCONVs[layer], layer, **kargs)
            self.autoedcoder.append(iconv)

    def _tie_weight(self, dConv, dlayer):
        karg = {}

        for k, v in self.tied_attr.items():
            if (dlayer in v.keys()): #(not v.get(dlayer, None) is None)
                if k in ['W']:
                    eConv = self.autoedcoder[v.get(dlayer)]
                    try:
                        dConv.lin_s.weight = eConv.lin_s.weight.transpose(0, 1)
                        dConv.lin_d.weight = eConv.lin_d.weight.transpose(0, 1)
                    except:
                        try:
                            dConv.lin_s.weight = nn.Parameter(eConv.lin_s.weight.transpose(0, 1))
                            dConv.lin_d.weight = nn.Parameter(eConv.lin_d.weight.transpose(0, 1))
                        except:
                            dConv.lin_s.data = nn.Parameter(eConv.lin_s.data.transpose(0, 1))
                            dConv.lin_d.data = nn.Parameter(eConv.lin_d.data.transpose(0, 1))
                if k in ['R']:
                    eConv = self.autoedcoder[v.get(dlayer)]
                    if not eConv.res_fc is None:
                        try:
                            dConv.res_fc.weight = eConv.res_fc.weight.transpose(0, 1)
                        except:
                            try:
                                dConv.res_fc.weight = nn.Parameter(eConv.res_fc.weight.transpose(0, 1))
                            except:
                                pass
                if k in ['A']:
                    eConv = self.autoedcoder[v.get(dlayer)]
                    karg['tie_alpha'] = eConv.tie_alpha

        return dConv, karg

    def forward(self,
                X: Union[Tensor, PairTensor],
                edge_index: Adj,
                **kargs
                ):

        # Encoder
        H = X
        # L = self.line_rese(X)
        for elayer in range(self.e_layers):
            eConv = self.autoedcoder[elayer]
            kargsn = { k: kargs.get(k, v) for k,v in self.forwargs[elayer].items() }
            H = eConv(H, edge_index, **kargsn)
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
            kargsn = { k: kargs.get(k, v) for k,v in self.forwargs[dlayer].items() }
            kargsn.update(karg)
            H = dConv(H, edge_index, **kargsn)

            if dlayer != (self.n_layers - 1):
                H = self.hidden_act(H)
                # H = F.dropout(H, p=self.dropout, training=self.training)
            else:
                H = self.final_act(H)

        self.X_ = H
        return self.H, self.X_ #, self.attention