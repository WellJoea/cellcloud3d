#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : Untitled-1
* @Author  : Wei Zhou                                     *
* @Date    : 2023/11/26 16:31:48                          *
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
from torch_geometric.utils import remove_self_loops

from ._GATEconv import GATEConv
from ._utilis import seed_torch
from ._nnalign  import nnalign

from tqdm import tqdm
class GRINA():
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
              bias=False,
              add_self_loops = True,
              share_weights = True,

              lr=1e-4, 
              n_epochs=500,
              p_epoch=None,
              e_epochs=0,
              u_epoch=0,
              Lambda = 0,
              pGamma =0,
              nGamma =0,
              gat_temp = 1,
              loss_temp = 0.5,
              iLambda = 0,

              weight_decay = 1e-4,
              gradient_clipping = 5,
              device=None,
              edge_attr=None,
              tied_attr={'node_weight': 'all', 'attention':'all'},
              seed=491001,

              position = None,
              groups = None,
              edge_nn = 15,
              p_nn = 15,
              n_nn = 10,
              root=None,
              regist_pair=None,
              drawmatch=True,

              knn_method='hnsw',
              reg_method = 'rigid', 
              reg_nn=6,
              CIs = 0.93, 
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
                    add_self_loops = add_self_loops,
                    share_weights = share_weights,
                    gattype=gattype,
                    tied_attr=tied_attr,
                    gat_temp=gat_temp,
                    **kargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=lr, 
                                     weight_decay=weight_decay)
        loss_list = []
        for epoch in tqdm(range(1, n_epochs + 1)):
            model.train()
            optimizer.zero_grad()

            H, X_, A = model(X, edge_index, edge_attr = edge_attr)
            loss = model.loss_gae(X, X_, H, edge_index.detach(), Lambda=Lambda)

            loss_list.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if epoch % p_epoch == 0:
                print(f'total loss: {loss}')

        if e_epochs>0:
            with torch.no_grad():
                H, _, _= model(X, edge_index)
            u_epoch = u_epoch or 50
            if not position is None:
                new_pos = position.clone().numpy()
            #for epoch in tqdm(range(n_epochs, n_epochs + e_epochs)):
            for epoch in tqdm(range(e_epochs)):
                if epoch % u_epoch == 0 or epoch == n_epochs:
                    if position is None:
                        pmnns = GRINA.pnnpairs(H.detach().numpy(), 
                                            groups, 
                                            root=None, 
                                            regist_pair=None, 
                                            full_pair=True,
                                            knn_method=knn_method, 
                                            edge_nn=p_nn, 
                                            cross=True,
                                            set_ef=p_nn+5)
                        nmnns = GRINA.nnnself(H.detach().numpy(), 
                                              groups, 
                                              root=0,
                                              kns=n_nn, 
                                              seed = [epoch, seed], 
                                              exclude_edge_index = list(map(tuple, edge_index.detach().numpy().T)))

                        # nmnns = GRINA.nnnhself(H.detach().numpy(), 
                        #                       groups, 
                        #                       root=0,
                        #                       kns = n_nn, 
                        #                       #seed = None, 
                        #                       exclude_edge_index = edge_index.detach().numpy())
                        print(nmnns.shape, pmnns.shape)
                        edge_mnn = edge_index
                        edge_mnn = edge_mnn.to(device)
                        exclude_edge_index = None
                    else:
                        new_pos, tforms, pmnns, nmnns= GRINA.rmnnalign(new_pos, 
                                                                groups,
                                                                H.detach().numpy(), 
                                                        root=root, regist_pair=regist_pair, 
                                                        knn_method=knn_method,
                                                        edge_nn=edge_nn,
                                                        reg_method = reg_method,
                                                        reg_nn=reg_nn,
                                                        CIs = CIs,
                                                        drawmatch=drawmatch, 
                                                        line_width=line_width,
                                                        line_alpha=line_alpha,
                                                        line_limit=line_limit)
                        edge_mnn = torch.concatenate([edge_index, pmnns], axis=1)
                        exclude_edge_index = edge_mnn
                        edge_mnn = edge_mnn.to(device)
                        nmnns = None

                model.train()
                optimizer.zero_grad()
                H, X_, A = model(X, edge_mnn, edge_attr = edge_attr,)
                # loss = model.loss_gae(X, X_, H, edge_index.detach(),
                #                       Lambda=iLambda, pGamma = pGamma, nGamma = nGamma,
                #                         pos_edge_index=pmnns, 
                #                         neg_edge_index=nmnns,
                #                         exclude_edge_index=exclude_edge_index)
                loss = model.loss_mse(X, X_) + model.loss_contrast(H, pmnns, nmnns,
                                                                    temperature=loss_temp, 
                                                                    Lambda= pGamma)
                loss_list.append(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                if epoch % p_epoch == 0:
                    print(f'total loss: {loss}')

        if not position is None:
            new_pos, tforms, _,_ = GRINA.rmnnalign(position.clone().numpy(), 
                                                    groups,
                                                    H.detach().numpy(), 
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

        if self.save_latent:
            self.latent = H
        if self.save_alpha:
            self.alpha = A
        if self.save_x:
            self.X_ = X_
        self.loss = loss_list
        if self.save_model:
            self.model = model
        else:
            return model

    def infer(self, X, edge_index, edge_attr=None,  model = None):
        model = self.model if model is None else model
        model.eval()
        H, X_, A = model(X, edge_index, edge_attr=edge_attr)
        loss = model.loss_gae(X, X_, H, edge_index)
        print(f'infer loss: {loss}')
        return H, X_,A

    @staticmethod
    def pnnpairs(hData, groups, root=None, regist_pair=None, full_pair=False,
                   knn_method='hnsw',cross=True, edge_nn=15, set_ef=50, **kargs):
        mnnk = nnalign()
        mnnk.build(hData, 
                groups, 
                hData=None,
                method=knn_method,
                root=root,
                regist_pair=regist_pair,
                full_pair=full_pair)
        mnn_idx = mnnk.egdemnn(knn=edge_nn, cross=cross, 
                               return_dist=False,
                               set_ef=set_ef, **kargs)
        mnn_idx = torch.tensor(mnn_idx.T, dtype=torch.long)
        return mnn_idx

    @staticmethod
    def nnnself(hData, groups, root=0, kns=10, seed = 491001, exclude_edge_index = None):
        nnnk = nnalign()
        nnnk.build(hData, 
                groups, 
                hData=None,
                root=root)
        nnn_idx = nnnk.negative_self(kns=kns+1, seed = seed, 
                                     exclude_edge_index = exclude_edge_index)
        nnn_idx, _ = remove_self_loops(torch.tensor(nnn_idx.T, dtype=torch.long))
        return nnn_idx

    @staticmethod
    def nnnhself(hData, groups, root=0, kns=None, seed = 491001, exclude_edge_index = None):
        nnnk = nnalign()
        nnnk.build(hData, 
                groups, 
                hData=None,
                root=root)
        nnn_idx = nnnk.negative_hself(exclude_edge_index, kns=kns, seed = seed)
        nnn_idx, _ = remove_self_loops(torch.tensor(nnn_idx, dtype=torch.long))
        return nnn_idx

    @staticmethod
    def rmnnalign(position, groups, hData, root=None, 
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
        pmnns = mnnk.egdemnn(knn=edge_nn, cross=True, return_dist=False)
        nmnns = mnnk.egdemnn(knn=edge_nn, cross=True, return_dist=False, reverse=True)
        pmnns = torch.tensor(pmnns.T, dtype=torch.long)
        nmnns = torch.tensor(nmnns.T, dtype=torch.long)
        return new_pos, tforms, pmnns, nmnns