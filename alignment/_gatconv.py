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
# from torch.nn.utils import weight_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
    Size,
)

from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax
)
from torch_geometric.utils.sparse import set_sparse_value
from ._utilis import ssoftmax, glorot, Activation

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
        dropout (float, optional): Dropout probability of the normalized
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
        dropout: float = 0.0,
        sift: float = 0.0, 
        gat_temp: float = 1,

        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = False,
        aslinear: bool = False,
        share_weights: bool = False,
        weight_norml2: bool = False,
        residual_norml2: bool = False,
        residual: bool = False,
        active: Optional[str] = None,

        init_weights: bool = True,
        init_gain: Optional[float] = None,
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
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.residual = residual
        self.init_weights = init_weights
        self.init_gain = init_gain
        self.weight_norml2 = weight_norml2
        self.residual_norml2 = residual_norml2
        self.bias = bias
        self.aslinear = aslinear
        self.dense_alpha = dense_alpha
        self.sift=sift
        self.gat_temp=gat_temp
        self.active = active
        if isinstance(in_channels, int):
            in_channell, in_channelr = in_channels, in_channels
        else:
            in_channell, in_channelr = in_channels[:2]

        # self.lin_s = Linear(in_channell, heads * out_channels,
        #                         bias=bias, weight_initializer='glorot')
        self.lin_s = nn.Linear(in_channell, heads * out_channels, bias=bias)
        if self.weight_norml2:
            self.lin_s = weight_norm(self.lin_s, name='weight', dim=0)
        if share_weights:
            self.lin_d = self.lin_s
        else:
            # self.lin_d = Linear(in_channelr, heads * out_channels,
            #             bias=bias, weight_initializer='glorot')
            self.lin_d = nn.Linear(in_channelr, heads * out_channels, bias=bias)
            if self.weight_norml2:
                self.lin_d = weight_norm(self.lin_d, name='weight', dim=0)

        self.att_s = Parameter(torch.empty(1, heads, out_channels))
        self.att_d = Parameter(torch.empty(1, heads, out_channels))

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
                self.Bias = Parameter(torch.empty(heads * out_channels))
            elif not concat:
                self.Bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('Bias', None)

        self._alpha = None
        if self.init_weights:
            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        gain =  self.init_gain  or nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.lin_s.weight, gain=gain)
        if not self.lin_s.bias is None:
            zeros(self.lin_s.bias)

        nn.init.xavier_normal_(self.lin_d.weight, gain=gain)
        if not self.lin_d.bias is None:
            zeros(self.lin_d.bias)

        if self.res_fc is not None and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if not self.res_fc.bias is None:
                zeros(self.res_fc.bias)

        if self.lin_edge is not None:
            nn.init.xavier_normal_(self.lin_edge.weight, gain=gain)
            if not self.lin_edge.bias is None:
                zeros(self.lin_s.bias)

        nn.init.xavier_normal_(self.att_s.data, gain=gain)
        nn.init.xavier_normal_(self.att_d.data, gain=gain)
        zeros(self.Bias)
        glorot(self.att_edge)

    def reset_parameters1(self):
        super().reset_parameters()
        self.lin_s.reset_parameters()
        self.lin_d.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res_fc is not None:
            self.res_fc.reset_parameters()
        glorot(self.att_s)
        glorot(self.att_d)
        glorot(self.att_edge)
        zeros(self.Bias)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'concat={self.concat}, share_weights={self.share_weights}, '
                f'bias={self.bias}, '
                f'residual={self.residual})')

class sGATConv(gating):
    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)
        self.activation = Activation(self.active or 'leaky_relu', negative_slope=self.negative_slope)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        tie_alpha: OptTensor = None,
        size: Size = None,
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
        assert x.dim() == 2
        x_s = self.lin_s(x).view(-1, H, C)
        if self.share_weights:
            x_d = x_s
        else:
            x_d = self.lin_d(x).view(-1, H, C)

        assert x_s is not None
        assert x_d is not None
        num_nodes = size or x_s.size(0)
        
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                if x_d is not None:
                    num_nodes = min(num_nodes, x_d.size(0))
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
            return x_s.mean(dim=1)

        if tie_alpha is None:
            alpha_s = (x_s * self.att_s).sum(dim=-1)
            alpha_d = (x_d * self.att_d).sum(dim=-1)
            alpha =(alpha_s, alpha_d)
            self.tie_alpha = alpha
        else:
            alpha = tie_alpha

        out = self.propagate(edge_index, 
                            x = (x_s, x_d),
                            alpha =(alpha[0], alpha[1]),
                            edge_attr=edge_attr,
                            edge_weight=edge_weight,
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

        if self.Bias is not None:
            out = out + self.Bias

        return out

    def message(self, x_j: Tensor, x_i: Tensor, 
                alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor,
                edge_weight: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j + alpha_i
        if not edge_weight is None:
            alpha = alpha * edge_weight

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            edge_attr = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + edge_attr

        alpha = self.activation(alpha)
        alpha = ssoftmax(alpha, index, ptr, size_i, sift=self.sift, temp=self.gat_temp)
        self.alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

class edGATConv(sGATConv):
    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)
        self.activation = Activation(self.active or 'sigmoid', negative_slope=self.negative_slope)

class dGATConv(gating):
    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)
        self.activation = Activation(self.active or 'leaky_relu', negative_slope=self.negative_slope)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        tie_alpha: OptTensor = None,
        size: Size = None,
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
            x_s = self.lin_s(x).view(-1, H, C)
            if self.share_weights:
                x_d = x_s
            else:
                x_d = self.lin_d(x).view(-1, H, C)
        else:
            x_s, x_d = x[0], x[1]
            assert x[0].dim() == 2
            x_s = self.lin_s(x_s).view(-1, H, C)
            if x_d is not None:
                x_d = self.lin_d(x_d).view(-1, H, C)

        assert x_s is not None
        assert x_d is not None
        num_nodes = size or x_s.size(0)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                if x_d is not None:
                    num_nodes = min(num_nodes, x_d.size(0))
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
            return x_s.mean(dim=1)

        out = self.propagate(edge_index, 
                            x = (x_s, x_d),
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

        if self.Bias is not None:
            out = out + self.Bias
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

            alpha = self.activation(alpha)
            alpha = (alpha * self.att_s).sum(dim=-1)
            # if self.dense_alpha:
            #     alpha = torch.tanh(alpha)
            alpha = ssoftmax(alpha, index, ptr, size_i, sift=self.sift, temp=self.gat_temp)

        self.alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
