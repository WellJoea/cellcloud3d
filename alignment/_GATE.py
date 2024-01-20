#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _GATE.py
* @Author  : Wei Zhou                                     *
* @Date    : 2023/11/15 21:51:57                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

import torch
from cellcloud3d.alignment._GATEconv import GATEConv
from ._utilis import seed_torch
from tqdm import tqdm
from ..plotting import parahist

import torch_geometric.utils as tgu
# from torch_geometric.utils import (
#     add_self_loops,
#     is_torch_sparse_tensor,
#     remove_self_loops,
#     softmax
# )


class GATE():
    def __init__(self, save_model=True, save_latent=True, save_x=True, save_alpha=True):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha

    def train(self,
              X,
              edge_index,
              gattype='gatv2',
              hidden_dims=[512, 48], 
              Heads=1,
              Concats=False,

              add_self_loops = True,
              fill_value = 'mean',
              share_weights = True,
              gat_temp= 1,
              anchor_num = 5,
              lr=1e-4, 
              n_epochs=500,
              p_epoch=None,
              weight_decay = 1e-4,
              Lambda = 0,
              gradient_clipping = 5,
              device=None,
              edge_attr=None,
              layer_attr= None, #{0:{"residual":True}, 3:{'residual':True}},
              tied_attr= None, #{'W': 'all', 'R': 'all', 'A':[0]},
              seed=491001,
              show_plot = False,
              save_plot = None,
              **kargs
              ):
        hidden_dims = [X.shape[1]] + hidden_dims
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        p_epoch = p_epoch or max(n_epochs/10, 10)
        anchor_edeg_num = anchor_num * X.size(0)

        if add_self_loops:
            edge_index, edge_attr = tgu.remove_self_loops( edge_index, edge_attr)
            edge_index, edge_attr = tgu.add_self_loops(edge_index,
                                                    edge_attr,
                                                    fill_value=fill_value,
                                                    num_nodes=X.size(0))
        else:
            edge_index, edge_attr = tgu.remove_self_loops( edge_index, edge_attr)

        X = X.to(device)
        edge_index = edge_index.to(device)
        if not edge_attr is None:
            edge_attr = edge_attr.to(device)

        model = GATEConv(hidden_dims,
                    Heads=Heads,
                    Concats=Concats,
                    share_weights = share_weights,
                    gattype=gattype,
                    tied_attr=tied_attr,
                    layer_attr=layer_attr,
                    gat_temp=gat_temp,
                    **kargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=lr, 
                                     weight_decay=weight_decay)


        pbar = tqdm(range(n_epochs), colour='red')
        for char in pbar:
            model.train()
            optimizer.zero_grad()

            H, X_, A = model(X, edge_index, edge_attr = edge_attr)
            # print(A)
            # _, idx= torch.topk(A.squeeze(), anchor_edeg_num, dim=0)
            # edge_topk = edge_index.detach()[:,idx]
            # loss = model.loss_gae(X, X_, H, edge_index.detach(), Lambda=Lambda)
            # loss = model.loss_mse(X, X_) 
            loss = model.loss_gae(X, X_, H, edge_index, Lambda=Lambda)
            # loss = model.loss_mse(X, X_) + model.loss_soft_cosin(H, edge_index.detach(), temperature=1, Lambda=Lambda)
            # loss += model.loss_structure(H, edge_index.detach(), Lambda=Lambda)
            #loss += model.loss_recon(H, edge_index.detach(), pGamma=model.Lambda, nGamma=model.Lambda)
            #loss = model.loss_gae(X, X_, H, edge_index.detach(), Lambda=Lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            pbar.set_postfix(loss = f'{loss.item():.10f}')
            # if char % p_epoch == 0 or char==n_epochs-1:
            #     print(f'total loss: {loss}')
            #     if show_plot or save_plot:
            #         if save_plot:
            #             save_plot = f'{char}_{save_plot}'
            #         parahist(model, save=save_plot, show=show_plot)

        pbar.close()
        model.eval()
        H, X_, A = model(X, edge_index, edge_attr = edge_attr)

        self.Lambda = Lambda
        if self.save_latent:
            self.latent = H.detach().cpu().numpy()
        # if self.save_alpha:
        #     self.alpha = A.detach().cpu().numpy()
        if self.save_x:
            self.X_ = X_.detach().cpu().numpy()
        if self.save_model:
            self.model = model
        else:
            return model

    @torch.no_grad()
    def infer(self, X, edge_index, edge_attr=None,  model = None):
        model = self.model if model is None else model
        model.eval()
        H, X_, A = model(X, edge_index, edge_attr=edge_attr)
        # loss = model.loss_mse(X, X_)
        # if model.Lambda >0:
        #     loss_structure = model.loss_structure(H, edge_index, Lambda=model.Lambda)
        #     loss += loss_structure
        # else:
        #     loss_structure =0
        # print(f'infer loss: {loss}, structure loss: {loss_structure}')
        #(model.model.encoder[0].lin_l)
        return H, X_,A
