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

from tqdm import tqdm
torch.set_default_dtype(torch.float32)

class GRAL():
    def __init__(self, save_model=True, save_latent=True, save_x=True, 
                  save_alpha=True):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha

    def train(self,
              adata,
              basis='spatial',
              add_key = 'glatent',
              groupby = None,
              align = True,
              #images = None,
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

              norm_latent = True,
              root=None,
              regist_pair=None,
              step=1,
              drawmatch=True,
              edge_nn = 15, 
              knn_method='hnsw',
              reg_method = 'rigid', 
              reg_nn=6,
              CIs = 0.91, 
              line_width=0.5, 
              line_alpha=0.5,
              line_limit=None,
              line_sample=None,
              size=1, 
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

        adata.obsm['glatent'] = Hs
        adata.layers['deconv'] = Xs_

        if (not groupby is None) and align:
            groups = adata.obs[groupby]
            position = adata.obsm[basis]
            Hnorm = F.normalize(torch.FloatTensor(Hs), dim=1).numpy() if norm_latent else Hs
            self.rmnnalign(position, 
                            groups,
                            Hnorm, 
                            root=root, regist_pair=regist_pair, 
                            step = step,
                            knn_method=knn_method,
                            edge_nn=edge_nn,
                            reg_method = reg_method,
                            reg_nn=reg_nn,
                            CIs = CIs, drawmatch=drawmatch, 
                            line_width=line_width,
                            line_alpha=line_alpha,
                            line_limit=line_limit,
                            size = size,
                            line_sample=line_sample)

            adata.obsm['align'] = self.new_pos
            adata.uns['align_tforms'] = self.tforms
            print(f'finished: added to `.obsm["align"]`')
            print(f'          added to `.obsm["glatent"]`')
            print(f'          added to `.uns["align_tforms"]`')
            print(f'          added to `.layers["deconv"]`')

        self.Lambda = Lambda
        self.model = model

    def train_global(self,
              adata,
              basis='spatial',
              add_key = 'glatent',
              groupby = None,
              #images = None,
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

              norm_latent = True,
              samples = None,
              root=None,
              regist_pair=None,
              step=1,
              drawmatch=True,
              edge_nn = 15, 
              knn_method='hnsw',
              reg_method = 'rigid', 
              reg_nn=6,
              CIs = 0.91, 
              line_width=0.5, 
              line_alpha=0.5,
              line_limit=None,
              line_sample=None,
              size=1, 
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

              **kargs):
        print('computing GRAL...')
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        p_epoch = p_epoch or max(n_epochs/10, 10)

        node_num = adata.shape[0]
        position = adata.obsm[basis]

        posknn = GRAL.pknnpairs(adata,  groupby=groupby, basis=basis, n_neighbors=d_nn)

        edge_index = torch.LongTensor(adata.uns[f'{basis}_edges']['edges'])
        # edge_weight = torch.FloatTensor(np.array([adata.uns[f'{basis}_edges']['simiexp'],
        #                                           adata.uns[f'{basis}_edges']['simi']])).transpose(0,1)
        edge_weight = None
        edge_index, edge_weight = self_loop_check(edge_index, edge_weight,
                                                  add_self_loops=add_self_loops,
                                                    num_nodes=node_num )

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
            loss = model.loss_gae(data.x, X_, H, data.edge_index, Lambda=Lambda)
            # loss =  model.loss_mse(data.x, X_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            pbar.set_postfix(loss = f'{loss.item():.10f}')
        pbar.close()

        with torch.no_grad():
            H, X_ = model(data.x, data.edge_index, edge_attr = edge_weight)
        adata.obsm['glatent'] = H.detach().cpu().numpy()

        if not groupby is None:
            groups = adata.obs[groupby]
            Hnorm = F.normalize(H.detach(), dim=1) if norm_latent else H.detach()
            self.rmnnalign(position, 
                            groups,
                            Hnorm.cpu().numpy(), 
                            root=root, regist_pair=regist_pair, 
                            step = step,
                            knn_method=knn_method,
                            edge_nn=edge_nn,
                            reg_method = reg_method,
                            reg_nn=reg_nn,
                            CIs = CIs, drawmatch=drawmatch, 
                            line_width=line_width,
                            line_alpha=line_alpha,
                            line_limit=line_limit,
                            size = size,
                            line_sample=line_sample)

            adata.obsm['align'] = self.new_pos
            adata.uns['align_tforms'] = self.tforms
            adata.layers['deconv'] = X_.detach().cpu().numpy()
            print(f'finished: added to `.obsm["align"]`')
            print(f'          added to `.obsm["glatent"]`')
            print(f'          added to `.uns["align_tforms"]`')
            print(f'          added to `.layers["deconv"]`')

        self.Lambda = Lambda
        self.model = model

    @torch.no_grad()
    def infer(self, adata, groupby = None, basis='spatial', img_key="hires", img_add_key = 'tres', 
            add_self_loops = True, validate = True, infer_latent=True, infer_alignment=True,
            edge_attr=None,  model = None, device='cpu', verbose=2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        verbose and print('inferring GRAL...')
        if infer_latent:
            cellid, loader, _  = self.loadData(adata, 
                        groupby=groupby, 
                        basis=basis, 
                        add_self_loops=add_self_loops,
                        validate=validate)
            cellidx = np.argsort(cellid)
            model = self.model if model is None else model
            model = model.cpu().to(device)
            model.eval()

            Hs, Xs_, Loss  = [], [], []
            for batch in loader:
                batch_data = batch.cpu().to(device)
                H, X_ = model(batch_data.x, batch_data.edge_index, edge_attr = edge_attr)
                loss = model.loss_gae(batch_data.x, X_, H, batch_data.edge_index, Lambda=self.Lambda)
                Hs.append(H.detach().numpy())
                Xs_.append(X_.detach().numpy())
                Loss.append(loss.item())
            Hs = np.concatenate(Hs, axis=0)[cellidx,]
            Xs_ = np.concatenate(Xs_, axis=0)[cellidx,]
            adata.obsm['glatent'] = Hs
            adata.layers['deconv'] = Xs_
            verbose and  print(f'infer loss: {np.mean(Loss)}')
            verbose and  print(f'finished: added to `.obsm["glatent"]`')
            verbose and  print(f'          added to `.layers["deconv"]`')

        if infer_alignment and (not groupby is None):
            mtforms = dict(zip( self.Order, self.tforms))
            adata.obsm['align'] = np.zeros_like(adata.obsm[basis])

            groups = adata.obs[groupby]
            try:
                order = groups.cat.remove_unused_categories().cat.categories
            except:
                order = np.unique(groups)

            for igroup in order:
                idx = adata.obs[groupby] == igroup
                iadata = adata[idx,:]
                ipos = iadata.obsm[basis]
                itam = mtforms.get(igroup, np.eye(3))
                inew_pos = homotransform_point(ipos, itam, inverse=False)
                adata.obsm['align'][idx] = inew_pos
                if not igroup in adata.uns[basis].keys():
                    adata.uns[basis][igroup] = {}
                adata.uns[basis][igroup][f'{img_add_key}_postmat'] = itam

                try:
                    iimg = adata.uns[basis][igroup]['images'][img_key]
                    isf = adata.uns[basis][igroup]['scalefactors'].get(f'tissue_{img_key}_scalef',1)
                    itam_sf = rescale_tmat(itam, isf, trans_scale=True)

                    inew_img = homotransform(iimg, itam_sf)
                    adata.uns[basis][igroup]['images'][img_add_key] = inew_img
                    adata.uns[basis][igroup][f'{img_add_key}_imgtmat'] = itam_sf
                    adata.uns[basis][igroup]['scalefactors'][f'tissue_{img_add_key}_scalef'] = isf
                except:
                    if verbose >1: 
                        print(f'No image was found in `.uns[{basis}][{igroup}]["images"][{img_key}]`.')
                        print(f'pass images registration.')

            verbose and print(f'finished: added to `.obsm["align"]`')
            verbose and print(f'          added to `.uns["{basis}"][<group>]"]`')

    # @classmethod
    def rmnnalign(self, position, groups, hData, root=None, images=None,
                  regist_pair=None, full_pair=False, step=1,
                   knn_method='hnsw', edge_nn=15, size=1, 
                   reg_method = 'rigid', reg_nn=6,
                   CIs = 0.93, drawmatch=False,  line_sample=None,
                   line_width=0.5, line_alpha=0.5, line_limit=None,**kargs):
        mnnk = nnalign()
        mnnk.build(position, 
                    groups, 
                    hData=hData,
                    method=knn_method,
                    root=root,
                    regist_pair=regist_pair,
                    step=step,
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
                    line_sample=line_sample,
                    size=size,
                    line_alpha=line_alpha, **kargs)
        self.tforms = mnnk.tforms
        self.new_pos = mnnk.transform_points()
        # self.new_imgs = mnnk.transform_images(images)
        self.Order = mnnk.order

        # mnnk = nnalign()
        # mnnk.build(new_pos, 
        #         groups, 
        #         hData=None,
        #         method=knn_method,
        #         root=root,
        #         regist_pair=regist_pair,
        #         full_pair=full_pair)
        # pmnns = mnnk.pairmnn(knn=edge_nn, cross=True, return_dist=False)
        # nmnns = mnnk.pairmnn(knn=edge_nn, cross=True, return_dist=False, reverse=True)
        # pmnns = torch.tensor(pmnns.T, dtype=torch.long)
        # nmnns = torch.tensor(nmnns.T, dtype=torch.long)
        # return new_pos, tforms, pmnns, nmnns

    @staticmethod
    def loadData(adata, groupby=None, basis='spatial', add_self_loops=True, validate=True):
        # cellidx = adata.obs_names.values
        cellidx = np.arange(adata.shape[0])
        cellid = []
        data_list = []
        position = []
        if groupby is None:
            coords = adata.obsm[basis]
            edge_index = torch.LongTensor(adata.uns[f'{basis}_edges']['edges'])
            edge_index,_ = self_loop_check(edge_index,
                                        add_self_loops=add_self_loops,
                                        num_nodes=adata.shape[0] )
            X = torch.FloatTensor(adata.X.toarray() if issparse(adata.X) else adata.X)
            data = Data(x=X, edge_index=edge_index)
            cellid.append(cellidx)
            data_list.append(data)
            position.append(coords)
        else:
            try:
                groups = adata.obs[groupby].cat.remove_unused_categories().cat.categories
            except:
                groups = adata.obs[groupby].unique()
            for igrp in groups:
                idx = (adata.obs[groupby]==igrp)
                idata = adata[idx,:]
                icoord = idata.obsm[basis]

                edge_index = torch.LongTensor(adata.uns[f'{basis}_edges']['edges'][igrp])
                edge_index,_ = self_loop_check(edge_index,
                                            add_self_loops=add_self_loops,
                                            num_nodes=idata.shape[0] )
                X = torch.FloatTensor(idata.X.toarray() if issparse(idata.X) else idata.X)
                data = Data(x=X, edge_index=edge_index)
                            #edge_weight=edge_weight,
                            #edge_attr=torch.FloatTensor(dists)
                data.validate(raise_on_error=validate)
                cellid.append(cellidx[idx])
                data_list.append(data)
                position.append(icoord)
        cellid = np.concatenate(cellid, axis=0)
        loader = DataLoader(data_list, batch_size=1, shuffle=False)
        return cellid, loader, position

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
