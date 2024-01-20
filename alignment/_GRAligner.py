#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _GRAligner.py
* @Author  : Wei Zhou                                     *
* @Date    : 2023/12/01 05:43:30                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* Please give me feedback if you find any problems.       *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops

from ._GATEconv import GATEConv
from ._GATE import GATE
from ._utilis import seed_torch
from ._nnalign  import nnalign

from tqdm import tqdm

class GRAligner():
    def __init__(self, save_model=True, save_latent=True, save_x=True, save_alpha=True):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha

    def train(self,
              X,
              edge_index,
              position = None,
              groups = None,

              gattype='gatv2',
              hidden_dims=[512, 48], 
              Heads=1,
              Concats=False,
              bias=False,

              share_weights = True,
              weight_norml2=True,
              residual_norml2 =True,
              residual=False,
              edge_attr=None,

             tied_attr={'W': 'all', 
                        'R':'all', 
                        #'A': [0],
                        },
              layer_attr={},

              lr=1e-4, 
              n_epochs=500,
              p_epoch=None,
              Lambda = 0,

              gat_temp = 1,
              loss_temp = 0.5,

              weight_decay = 1e-4,
              gradient_clipping = 5,
              device=None,

              seed=491001,

              edge_nn = 15, 
              norm_latent=True,
              root=None,
              regist_pair=None,
              drawmatch=True,

              knn_method='hnsw',
              reg_method = 'rigid', 
              reg_nn=6,
              CIs = 0.91, 
              line_width=0.5, 
              line_alpha=0.5,
              line_limit=None,

              **kargs):

        hidden_dims = [X.shape[1]] + hidden_dims
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        p_epoch = p_epoch or max(n_epochs/10, 10)

        X = X.to(device)
        edge_index = edge_index.to(device)
        if not edge_attr is None:
            edge_attr = edge_attr.to(device)
        model = GATEConv(hidden_dims,
                    Heads=Heads,
                    Concats=Concats,
                    bias=bias,
                    # add_self_loops = add_self_loops,
                    share_weights = share_weights,
                    gattype=gattype,
                    tied_attr=tied_attr,
                    gat_temp=gat_temp,
                    layer_attr=layer_attr,
                    weight_norml2=weight_norml2,
                    residual_norml2=residual_norml2,
                    residual=residual,
                    **kargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=lr, 
                                     weight_decay=weight_decay)

        pbar = tqdm(range(n_epochs), total=n_epochs, colour='red')
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()

            H, X_, A = model(X, edge_index, edge_attr = edge_attr)
            loss = model.loss_gae(X, X_, H, edge_index.detach(), Lambda=Lambda)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            pbar.set_postfix(loss = f'{loss.item():.10f}')
            if epoch % p_epoch == 0 or epoch==n_epochs-1:
                print(f'total loss: {loss}')
        pbar.close()
        model.eval()

        H, X_, A = model(X, edge_index, edge_attr = edge_attr)

        if not position is None:
            Hnorm = F.normalize(H.detach(), dim=1) if norm_latent else H.detach()
            new_pos, tforms, _,_ = GRAligner.rmnnalign(position.clone().numpy(), 
                                                    groups,
                                                    Hnorm.cpu().numpy(), 
                                        root=root, regist_pair=regist_pair, 
                                        knn_method=knn_method,
                                        edge_nn=edge_nn,
                                        reg_method = reg_method,
                                        reg_nn=reg_nn,
                                        CIs = CIs, drawmatch=drawmatch, 
                                        line_width=line_width,
                                        line_alpha=line_alpha,
                                        line_limit=line_limit)
            self.new_pos = new_pos
            self.tforms = tforms

        self.Lambda = Lambda
        if self.save_latent:
            self.latent = H.detach().cpu().numpy()
        if self.save_alpha:
            self.alpha = A.detach().cpu().numpy()
        if self.save_x:
            self.X_ = X_.detach().cpu().numpy()
        if self.save_model:
            self.model = model
        else:
            return model

    @torch.no_grad()
    def infer(self, X, edge_index, edge_attr=None,  model = None, device=None):
        model = self.model if model is None else model
        device = next(model.parameters()).device if device is None else device
        X = X.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None

        model.eval()
        H, X_, A = model(X, edge_index, edge_attr=edge_attr, )
        loss = model.loss_gae(X, X_, H, edge_index, Lambda=self.Lambda)
        print(f'infer loss: {loss}')
        H = H.detach().cpu().numpy()
        X_ = X_.detach().cpu().numpy()
        A = A.detach().cpu().numpy()
        return H, X_,A

    @staticmethod
    def rmnnalign(position, groups, hData, root=None, 
                  images = None,
                  regist_pair=None, full_pair=False,
                   knn_method='hnsw', edge_nn=15,
                   reg_method = 'rigid', reg_nn=6,
                   CIs = 0.93, drawmatch=False, 
                   line_width=0.5, line_alpha=0.5, line_limit=None,**kargs):
        mnnk = nnalign()
        mnnk.build(position, 
                    groups, 
                    hData=hData,
                    method=knn_method,
                    root=root,
                    regist_pair=regist_pair,
                    full_pair=full_pair)
        mnnk.regist(knn=reg_nn,
                    method=reg_method,
                    cross=True, 
                    CIs = CIs, 
                    broadcast = True, 
                    drawmatch=drawmatch, 
                    fsize=4,
                    line_limit=line_limit,
                    line_width=line_width, 
                    line_alpha=line_alpha, **kargs)
        tforms = mnnk.tforms
        new_pos = mnnk.transform_points()

        mnnk = nnalign()
        mnnk.build(new_pos, 
                groups, 
                hData=None,
                method=knn_method,
                root=root,
                regist_pair=regist_pair,
                full_pair=full_pair)
        pmnns = mnnk.pairmnn(knn=edge_nn, cross=True, return_dist=False)
        nmnns = mnnk.pairmnn(knn=edge_nn, cross=True, return_dist=False, reverse=True)
        pmnns = torch.tensor(pmnns.T, dtype=torch.long)
        nmnns = torch.tensor(nmnns.T, dtype=torch.long)
        return new_pos, tforms, pmnns, nmnns