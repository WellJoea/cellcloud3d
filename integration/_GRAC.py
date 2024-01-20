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
import numpy as np

from scipy.sparse import issparse
from torch_geometric.data import Data
import torch_geometric.utils as tgu
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
import torch.nn.functional as F

from cellcloud3d.alignment._GATEconv import GATEConv
# from cellcloud3d.integration.STAligner._STALIGNER import STAligner as GATEConv
from cellcloud3d.alignment._GATE import GATE
from cellcloud3d.alignment._utilis import seed_torch
from cellcloud3d.alignment._nnalign  import nnalign
from cellcloud3d.tools._sswnn import SSWNN, findmatches
from cellcloud3d.tools._spatial_edges import trip_edges, spatial_edges


class GRAC():
    def __init__(self, save_model=True, save_latent=True, save_x=True, save_alpha=True):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha
        self.findmatches = findmatches

    def train(self,
              adata,
              basis='spatial',
              add_key = 'GRAC',
              groupby = None,
              gattype='gatv3',
              hidden_dims=[512, 48], 
              Heads=1,

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
              Beta = 0,
              Gamma = 0,
              gat_temp = 1,
              loss_temp = 1,
              
              norm_latent = False,
              device=None,
              seed=491001,
              
              root =None,
              regist_pair = None,
              full_pair=False,
              step=1,

              use_dpca = False,
              dpca_npca = 50,
              ckd_method ='annoy',
              m_neighbor= 6,
              e_neighbor = 30,
              s_neighbor = 30,
              o_neighbor = 30,
              lower = 0.01,
              upper = 0.9,
              line_width=0.5,
              line_alpha=0.5,
              line_limit=None,
              line_sample=None,
              drawmatch = False,
              point_size=1,
              n_nn = 10,

              **kargs):
        print('computing GRAC...')
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        p_epoch = p_epoch or max(n_epochs/10, 10)

        node_num = adata.shape[0]
        edge_index = torch.LongTensor(adata.uns[f'{basis}_edges']['edges'])
        # edge_weight = torch.FloatTensor(np.array([adata.uns[f'{basis}_edges']['simiexp'],
        #                                           adata.uns[f'{basis}_edges']['simi']])).transpose(0,1)
        edge_weight = None

        if add_self_loops:
            edge_index, edge_weight = tgu.remove_self_loops( edge_index, edge_weight)
            edge_index, edge_weight = tgu.add_self_loops(edge_index,
                                                    edge_weight,
                                                    fill_value='mean',
                                                    num_nodes=node_num)
        else:
            edge_index, edge_weight = tgu.remove_self_loops( edge_index, edge_weight)

        data = Data(x=torch.FloatTensor(adata.X.toarray() if issparse(adata.X) else adata.X),
                    edge_index=edge_index,
                    #edge_weight=edge_weight,
                    #edge_attr=torch.FloatTensor(dists),
                    )
        data.validate(raise_on_error=validate)
        del edge_index
        data = data.to(device)

        if not edge_attr is None:
            edge_attr = edge_attr.to(device)

        if not edge_weight is None:
            edge_weight = edge_weight.to(device)

        model = GATEConv(
                    [data.x.shape[1]] + hidden_dims,
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
            model.train()
            optimizer.zero_grad()
            H, X_ = model(data.x, data.edge_index, edge_attr = edge_weight)
            loss = model.loss_gae(data.x, X_, H, data.edge_index, Lambda=Lambda, Gamma=Gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            pbar.set_postfix(loss = f'{loss.item():.10f}')
        pbar.close()

        with torch.no_grad():
            H, _ = model(data.x, data.edge_index, edge_attr = edge_weight)
        adata.obsm['GATE'] = H.detach().cpu().numpy()

        if ( not groupby is None) and e_epochs>0:
            groups = adata.obs[groupby]
            position = adata.obsm[basis]
            u_epoch = u_epoch or 100
            pbar = tqdm(range(e_epochs), total=e_epochs, colour='blue')
            for epoch in pbar:
                if epoch % u_epoch == 0:
                    hData = H.detach().cpu()
                    Hnorm = F.normalize(torch.FloatTensor(hData), dim=1).numpy() if norm_latent else hData.numpy()
                    pmnns, ssnn_scr = self.findmatches(
                                            Hnorm, groups, position=position,
                                            ckd_method=ckd_method, 
                                            sp_method = 'sknn',
                                            use_dpca = use_dpca,
                                            dpca_npca = dpca_npca,
                                            root=root, regist_pair=regist_pair,
                                            full_pair=full_pair, step=step,
                                            m_neighbor=m_neighbor, 
                                            e_neighbor =e_neighbor, 
                                            s_neighbor =s_neighbor,
                                            o_neighbor =o_neighbor,
                                            lower = lower, upper = upper,
                                            point_size=point_size,
                                            drawmatch=drawmatch,  line_sample=None,
                                            line_width=line_width, line_alpha=line_alpha, 
                                            line_limit=line_limit)
                    pmnns = torch.tensor(pmnns, dtype=torch.long)
                    ssnn_scr = torch.tensor(ssnn_scr, dtype=torch.float32)

                    nmnns = GRAC.nnnself(Hnorm, 
                                            groups, 
                                            root=0,
                                            kns=n_nn, 
                                            seed = [epoch, seed], 
                                            exclude_edge_index = list(map(tuple, data.edge_index.detach().cpu().numpy().T)))

                    # print('pos.mnn: ', pmnns.shape, 'nev.sample: ', nmnns.shape)
                    # edge_mnn = edge_index
                    # edge_mnn = edge_mnn.to(device)

                    pmnns, nmnns, ssnn_scr =  pmnns.to(device), nmnns.to(device), ssnn_scr.to(device)

                model.train()
                optimizer.zero_grad()
                H, X_ = model(data.x, data.edge_index, edge_weight = edge_weight,)

                loss = model.loss_gae(data.x, X_, H, data.edge_index, Lambda=Lambda, Gamma=Gamma) + \
                       model.loss_contrast(H, pmnns, nmnns,
                                            temperature=loss_temp, 
                                            edge_weight = ssnn_scr,
                                            Beta= Beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.set_postfix(loss = f'{loss.item():.10f}; sswnn pairs:{pmnns.shape[1]}')
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

    @torch.no_grad()
    def infer(self, data, edge_attr=None,  model = None, device=None):
        model = self.model if model is None else model
        device = next(model.parameters()).device if device is None else device
        data = data.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None

        model.eval()
        H, X_ = model(data.x, data.edge_index, edge_attr=edge_attr)
        # loss = model.loss_gae(X, X_, H, edge_index)
        # print(f'infer loss: {loss}')
        return H, X_

    @staticmethod
    def mnnpairs(hData, groups, root=None, regist_pair=None, full_pair=False, 
                 step=1, keep_self=True,
                   knn_method='hnsw',cross=True, edge_nn=15, set_ef=50, **kargs):
        rqid = np.unique(groups)
        if len(rqid) ==1 :
            full_pair = False
            root=0
            regist_pair = [(0,0)]
            keep_self=True,
    
        mnnk = nnalign()
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
    def nnnself(hData, groups, root=0, kns=10, seed = 491001, exclude_edge_index = None):
        nnnk = SSWNN()
        nnnk.build(hData, groups,
                    splocs=None,
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