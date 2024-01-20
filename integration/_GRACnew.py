#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _GRAC.py
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

import numpy as np

from scipy.sparse import issparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cellcloud3d.alignment._GATEconv import GATEConv
from cellcloud3d.alignment._GATE import GATE
from cellcloud3d.alignment._utilis import seed_torch, self_loop_check
from cellcloud3d.alignment._nnalign  import nnalign
from cellcloud3d.tools._spatial_edges  import trip_edges, spatial_edges
from cellcloud3d.transform import homotransform_point, homotransform, rescale_tmat
from cellcloud3d.alignment._utilis import seed_torch, self_loop_check, loadData


from tqdm import tqdm
# torch.set_default_dtype(torch.float32)


class GRAC():
    def __init__(self, save_model=True, save_latent=True, save_x=True, save_alpha=True):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha
        self.loadData = loadData

    def train(self,
              adata,
              basis='spatial',
              add_key = 'GRAC',
              groupby = None,
              gattype='gatv3',
              hidden_dims=[512, 48], 
              Heads=1,
              samples=None,
              Concats=False,
              validate = True,
              add_self_loops = True,
              share_weights = True,
              weight_norml2=False,
              residual_norml2 =False,
              residual=False,
              bias=False,
              edge_attr=None,

              tied_attr={},
              layer_attr={},

              lr=1e-4, 
              weight_decay = 1e-4,
              gradient_clipping = 5,
              n_epochs=500,
              p_epoch=None,
              e_epochs=0,
              u_epoch=0,
              Lambda = 0,
              Beta =0,
              nGamma =0,
              gat_temp = 1,
              loss_temp = 1,

              device=None,
              seed=491001,
              anchor_nn = 5,
              p_nn = 15,
              d_nn = 50,
              n_nn = 10,

              root=None,
              regist_pair=None,
 
              full_pair=False,
              reg_step = 1,

              drawmatch=True,
              knn_method='hnsw',
            #   reg_method = 'rigid', 
            #   reg_nn=6,
            #   CIs = 0.92, 
            #   line_width=0.5, 
            #   line_alpha=0.5,
            #   line_limit=None,

              **kargs):
        print('computing GRAL...')
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        p_epoch = p_epoch or max(n_epochs/10, 10)

        cellid, loader, _  = self.loadData(adata,
                                            groupby=groupby,
                                            basis=basis,
                                            add_self_loops=add_self_loops,
                                            validate=validate)
        cellidx = np.argsort(cellid)

        model = GATEConv(
                    [adata.shape[1]] + hidden_dims,
                    Heads=Heads,
                    Concats=Concats,
                    bias=bias,
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
        for char in pbar:
            for batch in loader:
                model.train()
                optimizer.zero_grad()
                batch = batch.to(device)
                H, X_ = model(batch.x, batch.edge_index, edge_attr = None)

                loss = model.loss_gae(batch.x, X_, H, batch.edge_index, Lambda=Lambda)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                pbar.set_postfix(loss = f'{loss.item():.10f}')
        pbar.close()

        with torch.no_grad():
            Hs, Xs_  = [], []
            for batch in loader:
                batch_data = batch.cpu()
                H, X_ = model.cpu()(batch_data.x, batch_data.edge_index, edge_attr = None)
                Hs.append(H.detach().numpy())
                Xs_.append(X_.detach().numpy())
            Hs = np.concatenate(Hs, axis=0)[cellidx,]
            Xs_ = np.concatenate(Xs_, axis=0)[cellidx,]


        if ( not groupby is None) and e_epochs>0:
            groups = adata.obs[groupby]
            u_epoch = u_epoch or 100
            pbar = tqdm(range(e_epochs), total=e_epochs, colour='blue')
            for epoch in pbar:
                if epoch % u_epoch == 0:
                    hData = Hs.detach().cpu().numpy()
                    pmnns = GRAC.pnnpairs(hData, 
                                            groups, 
                                            root=root, 
                                            regist_pair=regist_pair, 
                                            full_pair=full_pair,
                                            step = reg_step,
                                            knn_method=knn_method, 
                                            edge_nn=p_nn, 
                                            cross=True,
                                            set_ef=p_nn+5)
                    pmnns = GRAC.psnnpairs(hData, pmnns, n_neighbors=30, min_samples= 25)

                    nmnns = GRAC.nnnself(hData, 
                                            groups, 
                                            root=0,
                                            kns=n_nn, 
                                            seed = [epoch, seed], 
                                            exclude_edge_index = list(map(tuple, data.edge_index.detach().cpu().numpy().T)))

                    print('pos.mnn: ', pmnns.shape, 'nev.sample: ', nmnns.shape)
                    # edge_mnn = edge_index
                    # edge_mnn = edge_mnn.to(device)

                    pmnns, nmnns =  pmnns.to(device), nmnns.to(device)

                model.train()
                optimizer.zero_grad()
                H, X_ = model(data.x, data.edge_index, edge_weight = edge_weight,)
                # loss = model.loss_mse(X, X_) + \
                loss = model.loss_gae(data.x, X_, H, data.edge_index, Lambda=Lambda) + \
                       model.loss_contrast(H, pmnns, nmnns,
                                            temperature=loss_temp, 
                                            Beta= Beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.set_postfix(loss = f'{loss.item():.10f}')
                # if epoch % p_epoch == 0 or epoch==e_epochs-1:
                #     print(f'total loss: {loss}')
            pbar.close()

        model.eval()
        H, X_ = model( data.x,  data.edge_index, edge_attr = edge_weight)

        self.Lambda = Lambda
        print(f'finished: added to `.obsm["{add_key}"]`')
        print(f'          added to `.layers["{add_key}"]`')
        adata.obsm[add_key] = H.detach().cpu().numpy()
        adata.layers[add_key] = X_.detach().cpu().numpy()

        if self.save_model:
            self.model = model
        else:
            return model

    # @torch.no_grad()
    # def infer(self, data, edge_attr=None,  model = None, device=None):
    #     model = self.model if model is None else model
    #     device = next(model.parameters()).device if device is None else device
    #     data = data.to(device)
    #     edge_attr = edge_attr.to(device) if edge_attr is not None else None

    #     model.eval()
    #     H, X_ = model(data.x, data.edge_index, edge_attr=edge_attr)
    #     # loss = model.loss_gae(X, X_, H, edge_index)
    #     # print(f'infer loss: {loss}')
    #     return H, X_

    @staticmethod
    def pnnpairs(hData, groups, root=None, regist_pair=None, full_pair=False, 
                 step=1, keep_self=True,
                   knn_method='hnsw',cross=True, edge_nn=15, set_ef=50, **kargs):
        rqid = np.unique(groups)
        if len(rqid) ==1 :
            full_pair = False
            root=0
            regist_pair = [(0,0)]
            keep_self=True,
    
        mnnk = sswnn()
        mnnk.build(hData, 
                groups, 
                hData=None,
                method=knn_method,
                root=root,
                regist_pair=regist_pair,
                step=step,
                full_pair=full_pair,
                keep_self=keep_self)
        mnn_idx = mnnk.pairmnn(knn=edge_nn, cross=cross, 
                               return_dist=False,
                               set_ef=set_ef, **kargs)
        # import matplotlib.pyplot as plt
        # radiu_trim = trip_edges(mnn_idx[:,2], filter = 'std2q')
        # plt.hist(mnn_idx[:,2], bins=100)
        # plt.axvline(radiu_trim, color='black', label=f'radius: {radiu_trim :.3f}')
        # plt.show()
        # mnn_idx = mnn_idx[mnn_idx[:,2]<=radiu_trim,:]
        mnn_idx = torch.tensor(mnn_idx[:,:2].T, dtype=torch.long)

        if len(rqid) ==1 :
            mnn_idx, _ = remove_self_loops(torch.tensor(mnn_idx, dtype=torch.long))
        return mnn_idx

    @staticmethod
    def psnnpairs(hData, mnn_idx, n_neighbors=10, min_samples=5, n_jobs=10):
        from cellcloud3d.integration._SNN import SNN
        # from cellcloud3d.SNN.snn import SNN
        print(mnn_idx.shape)
        eps = min(max(n_neighbors-5, 5), n_neighbors)
        snn = SNN(n_neighbors=n_neighbors, eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        # snn = SNN(30, 5, n_jobs=10)
        snn.fit(hData)
        label = snn.labels_
        label_s= label[mnn_idx[0].numpy()]
        label_r= label[mnn_idx[1].numpy()]
        cidx = label_r== label_s
        print(cidx.sum())
        return mnn_idx[:, cidx]

    @staticmethod
    def pknnpairs(adata, groupby=None, basis='spatial',  n_neighbors=30, infer_thred=0.98):
        spatial_edges(adata, 
                      groupby=groupby,
                      basis=basis,
                      add_key='pknn', 
                      radiu_trim='infer',
                      return_simi=False, 
                      return_simiexp = False,
                      n_neighbors=n_neighbors, 
                      remove_loop=False,
                      infer_thred=infer_thred,
                      show_hist=False,
                      verbose=False)
        
        return adata.uns['pknn']['edges']

    @staticmethod
    def nnnself(hData, groups, root=0, kns=10, seed = 491001, exclude_edge_index = None):
        nnnk = sswnn()
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
        nnnk = sswnn()
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
        mnnk = sswnn()
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

        mnnk = sswnn()
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