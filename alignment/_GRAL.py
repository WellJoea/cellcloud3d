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
from cellcloud3d.alignment._utilis import seed_torch, self_loop_check, loadData
from cellcloud3d.alignment._nnalign  import aligner
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
        self.loadData = loadData
        self.aligner = aligner

    def train(self,
              adata,
              basis='spatial',
              use_rep = 'X',
              add_embed = 'glatent',
              add_align = 'align',
              groupby = None,
              align = True,
              #images = None,
              gconvs='gatv3',
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

              norm_latent = False,
              root=None,
              regist_pair=None,
              step=1,
              drawmatch=True,

              use_dpca = True,
              dpca_npca = 60,
              ckd_method ='annoy',
              reg_method = 'rigid',
              m_neighbor= 6,
              e_neighbor = 30,
              s_neighbor = 30,
              o_neighbor = 30,
              lower = 0.05,
              CIs = 0.93,
              line_width=0.5,
              line_alpha=0.5,
              line_limit=None,
              line_sample=None,
              point_size=1,

              Lambda = 1,
              Gamma = 0,
              gat_temp = 1,
              device=None,
              seed=491001,

              **kargs):
        print('computing GRAL...')
        seed_torch(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        p_epoch = p_epoch or max(n_epochs/10, 10)

        cellid, loader, _  = self.loadData(adata,
                                            groupby=groupby,
                                            basis=basis,
                                            use_rep=use_rep,
                                            add_self_loops=add_self_loops,
                                            validate=validate)
        in_dim = loader.dataset[0].x.size(1)
        cellidx = np.argsort(cellid)
        model = GATEConv(
                    [in_dim] + hidden_dims,
                    Heads=Heads,
                    Concats=Concats,
                    bias=bias,
                    share_weights = share_weights,
                    gconvs=gconvs,
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

                loss = model.loss_gae(batch.x, X_, H, batch.edge_index, Lambda=Lambda, Gamma=Gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                pbar.set_postfix(loss = f'{loss.item():.8f}')
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

        adata.obsm[add_embed] = Hs
        # adata.layers['deconv'] = Xs_

        if (not groupby is None) and align:
            groups = adata.obs[groupby]
            position = adata.obsm[basis]
            Hnorm = F.normalize(torch.FloatTensor(Hs), dim=1).numpy() if norm_latent else Hs
            sargs = dict(
                        root=root,
                        regist_pair=regist_pair,
                        step = step,
                        m_neighbor = m_neighbor,
                        e_neighbor =e_neighbor,
                        s_neighbor =s_neighbor,
                        o_neighbor =o_neighbor,
                        use_dpca = use_dpca,
                        dpca_npca = dpca_npca,
                        lower = lower,
                        ckd_method=ckd_method,
                        reg_method = reg_method,
                        point_size = point_size,
                        CIs = CIs,
                        drawmatch=drawmatch,
                        line_sample=line_sample,
                        line_width=line_width,
                        line_alpha=line_alpha,
                        line_limit=line_limit)
            tforms, new_pos, matches, Order = self.aligner( Hnorm, groups, position, **sargs)

            self.Order = Order
            adata.obsm[add_align] = new_pos
            adata.uns[f'{add_align}_tforms'] = tforms
            adata.uns[f'{add_align}_matches'] = matches
            print(f'finished: added to `.obsm["{add_align}"]`')
            print(f'          added to `.obsm["{add_embed}"]`')
            print(f'          added to `.uns["{add_align}_tforms"]`')
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
