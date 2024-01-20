import numpy as np
import random
import os

from scipy.sparse import issparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional, Tuple, Union, Any
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.utils as tgu
from torch_geometric.utils import remove_self_loops

from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)

def loadData(adata, groupby=None, use_rep = 'X', basis='spatial', add_self_loops=True, validate=True):
    # cellidx = adata.obs_names.values
    cellidx = np.arange(adata.shape[0])
    cellid = []
    data_list = []
    position = []

    if use_rep == 'X':
        Xs = adata.X
    elif use_rep == 'raw':
        Xs = adata.raw.X
    else :
        Xs = adata.obsm[use_rep]

    if groupby is None:
        coords = adata.obsm[basis]
        edge_index = torch.LongTensor(adata.uns[f'{basis}_edges']['edges'])
        edge_index,_ = self_loop_check(edge_index,
                                    add_self_loops=add_self_loops,
                                    num_nodes=adata.shape[0] )
        Xs = torch.FloatTensor(Xs.toarray() if issparse(Xs) else Xs)
        data = Data(x=Xs, edge_index=edge_index)
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
            icoord = adata[idx,:].obsm[basis]
            iX  = Xs[idx]

            try:
                edge_index = adata.uns[f'{basis}_edges']['edges'][igrp]
            except:
                print(f"please check adata.uns['{basis}_edges']['edges'][{igrp}] or set 'merge_edges=False' in cc.tl.spatial_edges")
            edge_index = torch.LongTensor(edge_index)
            edge_index,_ = self_loop_check(edge_index,
                                            add_self_loops=add_self_loops,
                                            num_nodes=iX.shape[0] )
            iX = torch.FloatTensor(iX.toarray() if issparse(iX) else iX)
            data = Data(x=iX, edge_index=edge_index)
                        #edge_weight=edge_weight,
                        #edge_attr=torch.FloatTensor(dists)
            data.validate(raise_on_error=validate)
            cellid.append(cellidx[idx])
            data_list.append(data)
            position.append(icoord)
    cellid = np.concatenate(cellid, axis=0)
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    return cellid, loader, position

def self_loop_check(edge_index, edge_weight=None, num_nodes=None, fill_value='mean', add_self_loops=True):
    if add_self_loops:
        edge_index, edge_weight = tgu.remove_self_loops( edge_index, edge_weight)
        edge_index, edge_weight = tgu.add_self_loops(edge_index,
                                                edge_weight,
                                                fill_value=fill_value,
                                                num_nodes=num_nodes)
    else:
        edge_index, edge_weight = tgu.remove_self_loops( edge_index, edge_weight)
    return edge_index, edge_weight

def Activation(active, negative_slope=0.01):
    if active is None:
        return nn.Identity()
    elif active == 'relu':
        return nn.ReLU()
    elif active in ['leaky_relu', 'lrelu']:
        return nn.LeakyReLU(negative_slope)
    elif active == 'elu':
        return nn.ELU()
    elif active == 'selu':
        return nn.SELU()
    elif active == 'tanh':
        return torch.tanh
    elif active == 'sigmoid':
        return nn.Sigmoid()
    elif active == 'softmax':
        return nn.Softmax(dim=1)
    elif active == 'softplus':
        return nn.Softplus()
    elif active == 'srelu':
        return shiftedReLU.apply
    elif active == 'linear':
        return nn.Identity()
    else:
        return active

def seed_torch(seed=200504):
    if seed:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # try:
        #     torch.use_deterministic_algorithms(True)
        # except:
        #     pass

def glorot(value: Any, gain: float = 1.):
    if isinstance(value, Tensor):
        stdv = np.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-gain*stdv, gain*stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v, gain=gain)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v, gain=gain)

def ssoftmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    sift : Optional[float] = 0.,
    temp : Optional[float] = 1.,
    dim: int = 0,
) -> Tensor:

    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:
        >>> src = torch.tensor([1., 1., 1., 1.])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> ptr = torch.tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = ((src - src_max)/temp).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = (src - src_max.index_select(dim, index))/temp
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (sift + out_sum)

class shiftedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, ):
        out = inp
        out[out <= 0.05] = 0
        ctx.save_for_backward(out)
        # out = torch.zeros_like(inp) #.cuda()
        # out[inp <= 0.05] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inp <= 0 ] = 0
        return grad_input
