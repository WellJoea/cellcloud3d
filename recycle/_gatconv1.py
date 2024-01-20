#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _gatconv.py
* @Author  : Wei Zhou                                     *
* @Date    : 2023/09/15 21:47:51                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

from typing import Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax
)
from torch_geometric.utils.sparse import set_sparse_value
from ._utilis import ssoftmax, glorot

class gating(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}_{s}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)}
        \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_j
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_k
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        adropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:`\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}`.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        adropout: float = 0.0,
        sift: float = 0.0, 
        gat_temp: float = 1,

        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        aslinear: bool = False,
        share_weights: bool = False,
        weight_norml2: bool = True,
        residual_norml2: bool = True,
        residual: bool = False,
        gattype: Optional[str] = None,

        init_weights: bool = True,
        dense_alpha: bool = False,

        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.adropout = adropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.residual = residual
        self.gattype = gattype or 'gatv3'
        self.init_weights = init_weights
        self.weight_norml2 = weight_norml2
        self.residual_norml2 = residual_norml2
        self.aslinear = aslinear
        self.dense_alpha = dense_alpha
        self.sift=sift
        self.gat_temp=gat_temp
        if isinstance(in_channels, int):
            in_channell, in_channelr = in_channels, in_channels
        else:
            in_channell, in_channelr = in_channels[:2]

        # self.lin_l = Linear(in_channell, heads * out_channels,
        #                         bias=bias, weight_initializer='glorot')
        self.lin_l = nn.Linear(in_channell, heads * out_channels, bias=bias)
        if self.weight_norml2:
            self.lin_l = weight_norm(self.lin_l, name='weight', dim=0)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            # self.lin_r = Linear(in_channelr, heads * out_channels,
            #             bias=bias, weight_initializer='glorot')
            self.lin_r = nn.Linear(in_channelr, heads * out_channels, bias=bias)
            if self.weight_norml2:
                self.lin_r = weight_norm(self.lin_r, name='weight', dim=0)
 
        self.att_l = Parameter(torch.empty(1, heads, out_channels))
        self.att_r = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            # self.lin_edge = Linear(edge_dim, heads * out_channels, bias=bias,
            #                        weight_initializer='glorot')
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=bias)
            if self.weight_norml2:
                self.lin_edge = weight_norm(self.lin_edge, name='weight', dim=0)
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        self.has_linear_res = False
        if self.residual:
            if in_channell != heads * out_channels:
                self.has_linear_res = True
                self.res_fc = nn.Linear(in_channell, heads * out_channels, bias=bias)
                if self.residual_norml2:
                    self.res_fc = weight_norm(self.res_fc, name='weight', dim=0)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_parameter("res_fc", None)
        
        if bias and not self.has_linear_res:
            if concat:
                self.bias = Parameter(torch.empty(heads * out_channels))
            elif not concat:
                self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        if self.init_weights:
            self.reset_parameters()

        if self.gattype in ['gatv3']:
            self.active = torch.sigmoid
        elif self.dense_alpha:
            self.active = torch.tanh
        elif self.gattype in ['gat', 'gatv1']:
            self.active = nn.LeakyReLU(self.negative_slope)
        else:
            self.active = None

    def reset_parameters(self):
        super().reset_parameters()
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.lin_l.weight, gain=gain)
        if not self.lin_l.bias is None:
            zeros(self.lin_l.bias)

        nn.init.xavier_normal_(self.lin_r.weight, gain=gain)
        if not self.lin_r.bias is None:
            zeros(self.lin_r.bias)

        if self.res_fc is not None and isinstance(self.res_fc, nn.Linear):

            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if not self.res_fc.bias is None:
                zeros(self.res_fc.bias)

        if self.lin_edge is not None:
            # self.lin_edge.reset_parameters()
            nn.init.xavier_normal_(self.lin_edge.weight, gain=gain)
            if not self.lin_edge.bias is None:
                zeros(self.lin_l.bias)

        nn.init.xavier_normal_(self.att_l.data, gain=gain)
        nn.init.xavier_normal_(self.att_r.data, gain=gain)
        zeros(self.bias)
        glorot(self.att_edge)

    def reset_parameters1(self):
        super().reset_parameters()
        # self.lin_l.reset_parameters()
        # self.lin_r.reset_parameters()
        gain = 1 #nn.init.calculate_gain("relu")

        glorot(self.lin_l.weight, gain=gain)
        if not self.lin_l.bias is None:
            zeros(self.lin_l.bias)
        glorot(self.lin_r.weight, gain=gain)
        if not self.lin_r.bias is None:
            zeros(self.lin_r.bias)
        # nn.init.xavier_normal_(self.lin_r.weight, gain=2)

        if self.res_fc is not None:
            # self.lin_edge.reset_parameters()
            glorot(self.res_fc.weight, gain=gain)
            if not self.res_fc.bias is None:
                zeros(self.res_fc.bias)

        if self.lin_edge is not None:
            # self.lin_edge.reset_parameters()
            glorot(self.lin_edge.weight, gain=gain)
            if not self.lin_edge.bias is None:
                zeros(self.lin_l.bias)
        glorot(self.att_l, gain=gain)
        glorot(self.att_r, gain=gain)
        zeros(self.bias)
        glorot(self.att_edge)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'concat={self.concat}, share_weights={self.share_weights}, '
                f'gattype={self.gattype}, negative_slope={self.negative_slope}, '
                f'residual={self.residual}, add_self_loops={self.add_self_loops})')

class sGATConv(gating):
    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        tie_alpha: OptTensor = None,
    ):
        r"""Runs the forward pass of the module.
        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels
        S = x.size(0)
        assert x.dim() == 2
        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        if self.aslinear:
            self.alpha = None
            return x_l.mean(dim=1)

        if tie_alpha is None:
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
            alpha =(alpha_l, alpha_r)
            self.tie_alpha = alpha
        else:
            alpha = tie_alpha
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, 
                            x = (x_l, x_r),
                            alpha =(alpha[0], alpha[1]),
                            edge_attr=edge_attr,
                            size=None)

        alpha = self.alpha
        assert alpha is not None

        if self.residual:
            resi = self.res_fc(x).view(-1, H, C)
            out = out + resi

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, x_i: Tensor, 
                alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            edge_attr = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + edge_attr

        alpha = self.active(alpha)
        # alpha = ssoftmax(alpha, index, ptr, size_i, sift=self.sift, temp=self.gat_temp)
        alpha = softmax(alpha, index, ptr, size_i)
        self.alpha = alpha

        alpha = F.dropout(alpha, p=self.adropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

class dGATConv(gating):
    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        tie_alpha: OptTensor = None,
    ):
        r"""Runs the forward pass of the module.
        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        if self.aslinear:
            self.alpha = None
            return x_l.mean(dim=1)

        out = self.propagate(edge_index, 
                            x = (x_l, x_r),
                            edge_attr=edge_attr,
                            tie_alpha = tie_alpha,
                            size=None)
        alpha = self.tie_alpha = self.alpha
        assert alpha is not None

        if self.residual:
            resi = self.res_fc(x).view(-1, H, C)
            out = out + resi

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

        # if isinstance(edge_index, Tensor):
        #     if is_torch_sparse_tensor(edge_index):
        #         # TODO TorchScript requires to return a tuple
        #         adj = set_sparse_value(edge_index, alpha)
        #         return out, (adj, alpha)
        #     else:
        #         return out, (edge_index, alpha)
        # elif isinstance(edge_index, SparseTensor):
        #     return out, (edge_index.set_value(alpha, layout='coo'), alpha)

    def message(self, x_j: Tensor, x_i: Tensor, 
                edge_attr: OptTensor,
                tie_alpha: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if not tie_alpha is None:
            alpha = tie_alpha
        else:
            alpha = x_i + x_j
            if edge_attr is not None:
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)
                assert self.lin_edge is not None
                edge_attr = self.lin_edge(edge_attr)
                edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
                alpha = alpha + edge_attr

            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = (alpha * self.att_l).sum(dim=-1)
            # if self.dense_alpha:
            #     alpha = torch.tanh(alpha)
            #alpha = ssoftmax(alpha, index, ptr, size_i, sift=self.sift, temp=self.gat_temp)
            alpha = softmax(alpha, index, ptr, size_i)
    
        self.alpha = alpha
        alpha = F.dropout(alpha, p=self.adropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
