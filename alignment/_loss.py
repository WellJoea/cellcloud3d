#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _loss.py
* @Author  : Wei Zhou                                     *
* @Date    : 2023/11/27 16:50:26                          *
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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, batched_negative_sampling,scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

from ._utilis import ssoftmax

EPS = 1e-15
MAX_LOGSTD = 10

def record_funcv(func, name, value=None):
    if not hasattr(func, name):
        setattr(func,  name, value or 0)
    else:
        if value is None:
            value = getattr(func, name) or 0
            setattr(func, name, value)
        else:
            setattr(func, name, value)

def loss_gae(X, X_, H, edge_index, 
               Lambda=0, Gamma=0,
               pos_edge_index=None,
               neg_edge_index=None, 
               exclude_edge_index=None, 
               pGamma=0, nGamma=0):

    # record_funcv(loss_gae, 'Lambda', Lambda)
    # record_funcv(loss_gae, 'pGamma', pGamma)
    # record_funcv(loss_gae, 'nGamma', nGamma)

    return (loss_mse(X, X_) + 
            loss_structure(H, edge_index, Lambda=Lambda) + 
            loss_embedding(H, Gamma=Gamma) +
            loss_recon(H, pos_edge_index=pos_edge_index, 
                       neg_edge_index=neg_edge_index, 
                       exclude_edge_index=exclude_edge_index,
                       pGamma=pGamma,
                       nGamma=nGamma))

def loss_mse(X, X_):
    return F.mse_loss(X, X_)

def loss_embedding(embedding, Gamma=0):
    if (not Gamma):
        return 0
    else:
        loss = torch.square(embedding)
        loss = loss.mean() * Gamma
        return loss

def loss_cosine(H, edge_index, Lambda=1):
    if (not Lambda):
        return 0

    Hnorm = F.normalize(H, dim=1)
    L_emb = Hnorm[edge_index[0]]
    H_emb = Hnorm[edge_index[1]]

    similarity = - Lambda * (torch.sum(L_emb * H_emb, -1)).mean()
    return similarity

def loss_similar(H, edge_index, normal='logsigmoid', Lambda=1):
    #Hnorm = F.normalize(H, dim=1)
    #Hnorm = H/torch.mean(H, dim=1)
    L_emb = H[edge_index[0]]
    H_emb = H[edge_index[1]]

    # L_emb = H[edge_index[0]]
    # H_emb = H[edge_index[1]]
    similarity = torch.sum(L_emb * H_emb, -1)

    if normal is None:
        return Lambda * similarity
    elif normal == 'sigmoid':
        return Lambda * torch.sigmoid(similarity)
    elif normal == 'logsigmoid':
        return Lambda * torch.nn.LogSigmoid()(similarity)
    elif normal == 'neglogsigmoid':
        #return Lambda * torch.log(1-torch.sigmoid(similarity))
        return Lambda * (torch.nn.LogSigmoid()(similarity) -similarity)

def loss_similar_all(H, normal='logsigmoid', Lambda=1):
    adj = torch.matmul(H, H.t())

    if normal is None:
        return Lambda * adj
    elif normal == 'sigmoid':
        return Lambda * torch.sigmoid(adj)
    elif normal == 'logsigmoid':
        return Lambda * torch.nn.LogSigmoid()(adj)

def loss_structure(H, edge_index, Lambda=0.5):
    if (not Lambda):
        return 0
    else:
        structure = -loss_similar(H, edge_index, normal='logsigmoid', Lambda=Lambda).mean()
        return structure

def loss_recon(H, pos_edge_index=None, neg_edge_index=None, exclude_edge_index=None, 
               pGamma=1, nGamma=1 ):
    if  (pos_edge_index is None) or (not pGamma):
        pos_loss = 0
    else:
        pos_loss = -loss_similar(H, pos_edge_index, normal='logsigmoid', Lambda=pGamma).mean()

    if (neg_edge_index is False) or (not nGamma):
        neg_loss = 0
    else :
        if neg_edge_index is None:
            exclude_edge_index = pos_edge_index if exclude_edge_index is None else exclude_edge_index
            neg_edge_index = negative_sampling(exclude_edge_index, H.size(0))
        neg_loss = -loss_similar(H, neg_edge_index, normal='neglogsigmoid', Lambda=nGamma).mean()

    return pos_loss + neg_loss

def loss_cosin(H, edge_index):
    Hnorm = F.normalize(H, dim=1)
    L_emb = Hnorm[edge_index[0]]
    H_emb = Hnorm[edge_index[1]]
    return torch.sum(L_emb * H_emb, -1)

def loss_soft_cosin(H, edge_index, Lambda=1, temperature=1, sift=0):
    if not Lambda:
        return 0
    L_emb = H[edge_index[0]]
    H_emb = H[edge_index[1]]
    simi = torch.sum(L_emb * H_emb, -1)
    #simi = loss_cosin(H, edge_index)
    ssimi = ssoftmax(simi, edge_index[1], temp=temperature, sift=sift )

    loss = -Lambda * (torch.log(ssimi)).mean()
    return loss

def loss_soft_recon(H, pos_edge_index=None, neg_edge_index=None, exclude_edge_index=None, 
               pGamma=1, nGamma=1 ):
    if  (pos_edge_index is None) or (not pGamma):
        pos_loss = 0
    else:
        pos_loss = -loss_similar(H, pos_edge_index, normal='logsigmoid', Lambda=pGamma).mean()

    if (neg_edge_index is False) or (not nGamma):
        neg_loss = 0
    else :
        if neg_edge_index is None:
            exclude_edge_index = pos_edge_index if exclude_edge_index is None else exclude_edge_index
            neg_edge_index = negative_sampling(exclude_edge_index, H.size(0))
        neg_loss = -loss_similar(H, neg_edge_index, normal='neglogsigmoid', Lambda=nGamma).mean()

    return pos_loss + neg_loss

def loss_contrast(H, pos_edge_index, neg_edge_index, edge_weight = 1, num_node=None, temperature=0.5, Beta=1):
    if not Beta:
        return 0
    N = maybe_num_nodes( pos_edge_index[1], num_node or H.shape[0])

    pos_simi = loss_cosin(H, pos_edge_index)
    neg_simi = loss_cosin(H, neg_edge_index)

    # nominator = torch.exp((pos_simi-1) / temperature)
    # denominator = torch.exp((neg_simi-1) / temperature)
    nominator = torch.exp(pos_simi / temperature)
    denominator = torch.exp(neg_simi / temperature)

    neg_sum = scatter(denominator, neg_edge_index[1], 0, dim_size=N, reduce='sum') + 1e-16
    neg_sum = neg_sum.index_select(0, pos_edge_index[1])
    # neg_sum = neg_sum + nominator

    loss = -Beta * torch.log(nominator/neg_sum)
    loss = edge_weight * loss
    return loss.mean()

def loss_kl(mu, logstd):
    r"""Computes the KL loss, either for the passed arguments :obj:`mu`
    and :obj:`logstd`, or based on latent variables from last encoding.

    Args:
        mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
            set to :obj:`None`, uses the last computation of :math:`\mu`.
            (default: :obj:`None`)
        logstd (torch.Tensor, optional): The latent space for
            :math:`\log\sigma`.  If set to :obj:`None`, uses the last
            computation of :math:`\log\sigma^2`. (default: :obj:`None`)
    """
    logstd = logstd.clamp(max=MAX_LOGSTD)
    return -0.5 * torch.mean( torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device=None, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = torch.matmul(representations, representations.transpose(0,1))

        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs

        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
