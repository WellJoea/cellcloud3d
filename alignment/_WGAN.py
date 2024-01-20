from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from tqdm import tqdm
from typing import List, Optional, Union
import numpy as np

from cellcloud3d.alignment._utilis import seed_torch


class Generator(nn.Module):
    def __init__(self, indim, hidden_dims=1024, negative_slope=0.2):
        super().__init__()
        self.gnet = nn.Sequential(
            nn.Linear(indim, hidden_dims),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dims, indim),
        )
    def forward(self, Gemb):
        return self.gnet(Gemb)

class Discriminator(nn.Module):
    def __init__(self, indim, hidden_dims=512, negative_slope=0.2):
        super().__init__()
        self.dnet = nn.Sequential(
            nn.Linear(indim, hidden_dims),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dims, 1),
            # nn.Sigmoid()
        )
    def forward(self, Demb):
        return self.dnet(Demb).view(-1)

def gradient_penatly(D, xr, xf, device):
    t = torch.rand(xr.size(0), 1).to(device)
    t = t.expand_as(xr)
    mid = t * xr + (1-t)*xf
    mid.requires_grad_()
    pred = D(mid)
    grads = torch.autograd.grad(outputs=pred, inputs=mid,
                           grad_outputs=torch.ones_like(pred),
                           create_graph=True,
                           retain_graph=True, 
                           only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) -1 , 2).mean()
    return gp

def WGAN(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    # cc.ag.seed_torch(200504)
    indim = x.shape[1]
    G = Generator(indim).to(device)
    D = Discriminator(indim).to(device)
    print(G)
    print(D)
    optimizer_G = torch.optim.Adam(G.parameters(),
                                    lr=1e-4,
                                    betas = (0.5, 0.999),
                                    weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(D.parameters(),
                                    lr=1e-4,
                                    betas = (0.5, 0.999),
                                    weight_decay=1e-4)

    x = x.to(device)
    y = y.to(device)
    for epoch in range(1000):
        for _ in range(5):
            zr = D(x)
            lossr = -zr.mean()

            # z = torch.randn_like(x).to(device)
            xf = G(y).detach()
            zp = D(xf)
            lossf = zp.mean()

            gp = gradient_penatly(D, x, xf.detach())
            loss_D = lossr + lossf + gp
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
        z = torch.randn_like(x).to(device)
        xf = G(z)
        predf = D(xf) #.detach()
        loss_G = -predf.mean()
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

def wgan(ref, quary, device=None,
        anchor_scale = 0.8,
        n_epochs =100,
        seed=491001):
    r"""
    Data reconstruction network training strategy
    
    Parameters
    ----------
    recon_models
        list of reconstruction model
    optimizer_recons
        list of reconstruction optimizer
    embds
        list of LGCN embd
    features
        list of rae node features
    batch_d_per_iter
        WGAN train iter numbers
    """
    seed_torch(seed)
    embd0, embd1 = (ref, quary)
    embd0 = torch.FloatTensor(embd0)
    embd1 = torch.FloatTensor(embd1)
    anchor_size = int(np.ceil(min(embd0.size(0), embd1.size(0))*anchor_scale))
    indim = embd0.size(1)
 
    batch_d_per_iter:Optional[int]=5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)

    embd0 = embd0.to(device)
    embd1 = embd1.to(device)
    G = Generator(indim).to(device)
    D = Discriminator(indim).to(device)

    optimizer_G = torch.optim.Adam(G.parameters(),
                                    lr=1e-4,
                                    betas = (0.5, 0.999),
                                    weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(D.parameters(),
                                    lr=1e-4,
                                    betas = (0.5, 0.999),
                                    weight_decay=1e-4)

    pbar = tqdm(range(n_epochs), total=n_epochs, colour='red')
    for char in pbar:
        for j in range(batch_d_per_iter):
            zr = D(embd0)
            zp = D(G(embd1).detach())

            anchor1 = zp.argsort(descending=True)[: anchor_size]
            anchor0 = zr.argsort(descending=False)[: anchor_size]
            embd0_anchor = embd0[anchor0, :] #.clone().detach()
            embd1_anchor = embd1[anchor1, :] #.clone().detach()

            optimizer_D.zero_grad()
            gp = gradient_penatly(D, embd0[anchor0, :], embd1[anchor1, :].clone().detach(), device)
            loss_D = -torch.mean(D(embd0_anchor)) + torch.mean(D(G(embd1_anchor))) +  gp
            loss_D.backward()
            optimizer_D.step()
            for p in D.parameters():
                p.data.clamp_(-0.1, 0.1)

        zr = D(embd0)
        zp = D(G(embd1))

        anchor1 = zp.argsort(descending=True)[: anchor_size]
        # anchor0 = zr.argsort(descending=False)[: anchor_size]
        # embd0_anchor = embd0[anchor0, :]
        embd1_anchor = embd1[anchor1, :]
        loss_G = -torch.mean(D(G(embd1_anchor)))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        pbar.set_postfix(loss = f'loss_D: {loss_D.item():.6f}, loss_G: {loss_G.item():.6f}')

    pbar.close()
    return G,D

# cc.ag.seed_torch(200504)
# embd0 = torch.FloatTensor(X)
# embd1 = torch.FloatTensor(Y)
# G,D = wgan(embd0, embd1, device=None, n_epochs =1000)