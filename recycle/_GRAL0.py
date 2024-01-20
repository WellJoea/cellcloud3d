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

from cellcloud3d.alignment._GATEconv import GATEConv
# from cellcloud3d.integration.STAligner._STALIGNER import STAligner as GATEConv

from cellcloud3d.alignment._GATE import GATE
from cellcloud3d.alignment._utilis import seed_torch, self_loop_check
from cellcloud3d.alignment._nnalign  import nnalign
from cellcloud3d.alignment._neighbors  import trip_edges, spatial_edges
from cellcloud3d.transform import homotransform_point, homotransform

from tqdm import tqdm

class GRAL():
    def __init__(self, save_model=True, save_latent=True, save_x=True, save_alpha=True):
        self.save_model = save_model
        self.save_latent = save_latent
        self.save_x = save_x
        self.save_alpha = save_alpha

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
            add_self_loops = True, validate = True,
            edge_attr=None,  model = None, device=None):
        print('inferring GRAL...')
        edge_index = adata.uns[f'{basis}_edges'].get('edges', None)
        if not edge_index is None:
            edge_index = torch.LongTensor(edge_index)
            edge_index, edge_attr = self_loop_check(edge_index, edge_attr,
                                                    add_self_loops=add_self_loops,
                                                        num_nodes=adata.shape[0])

            data = Data(x=torch.FloatTensor(adata.X.toarray() if issparse(adata.X) else adata.X),
                        edge_index=edge_index,
                        edge_attr=edge_attr)
            data.validate(raise_on_error=validate)
            del edge_index
            del edge_attr
            data = data.to(device)

            model = self.model if model is None else model
            model.eval()
            device = next(model.parameters()).device if device is None else device

            H, X_ = model(data.x, data.edge_index, edge_attr = data.edge_attr)
            loss = model.loss_gae(data.x, X_, H, data.edge_index, Lambda=self.Lambda)
            print(f'infer loss: {loss}')

            adata.obsm['glatent'] = H.detach().cpu().numpy()
            adata.layers['deconv'] = X_.detach().cpu().numpy()

            print(f'finished: added to `.obsm["glatent"]`')
            print(f'          added to `.layers["deconv"]`')

        mtforms = dict(zip( self.Order, self.tforms))
        adata.obsm['align'] = np.zeros_like(adata.obsm[basis])
        if not groupby is None:
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

                iimg = adata.uns[basis][igroup]['images'][img_key]
                isf = adata.uns[basis][igroup]['scalefactors'].get(f'tissue_{img_key}_scalef',1)

                # ih, iw = iimg.shape[:2]
                # ipos_con = np.array([[0, 0], [0, iw], [ih,0], [ih, iw]])/isf
                # ipos_con = np.array([[0, 0], [0, ih], [iw,0], [iw, ih]])/isf
                # ipos_range = np.vstack([ipos, ipos_con])

                scale_l  = np.eye(3)
                scale_l[[0,1], [0,1]] = isf
                scale_l[:2,2] = isf
                scale_r = np.eye(3)
                scale_r[[0,1], [0,1]] = 1/isf
                itam_sf = scale_l @ itam @ scale_r

                inew_pos = homotransform_point(ipos, itam, inverse=False)
                inew_img = homotransform(iimg, itam_sf)

                adata.obsm['align'][idx] = inew_pos
                adata.uns[basis][igroup][f'{img_add_key}_tmat'] = itam
                adata.uns[basis][igroup]['images'][img_add_key] = inew_img
                adata.uns[basis][igroup]['images'][img_add_key]
                adata.uns[basis][igroup]['scalefactors'][f'tissue_{img_add_key}_scalef'] = isf

            print(f'finished: added to `.obsm["align"]`')
            print(f'          added to `.uns["{basis}"][*]["images"]["{img_add_key}"]`')

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
