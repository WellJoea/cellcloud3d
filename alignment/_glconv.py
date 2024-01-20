from typing import Tuple, Union

from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
# from torch.nn.utils import weight_norm

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import (
    scatter,
    add_self_loops,
    remove_self_loops,
)

from torch_geometric.utils import spmm

class kGNN(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot \mathbf{x}_j

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
        add_self_loops: bool = False,
        fill_value: Union[float, Tensor, str] = 'mean',

        weight_norml2: bool = False,
        degree_norml: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.add_self_loops = add_self_loops
        self.bias = bias
        self.weight_norml2 = weight_norml2
        self.degree_norml = degree_norml
        self.fill_value = fill_value

        self.lin_s = Linear(in_channels[0], out_channels, bias=self.bias)
        self.lin_d = Linear(in_channels[1], out_channels, bias=False)

        if self.weight_norml2:
            self.lin_s = weight_norm(self.lin_s, name='weight', dim=0)
            self.lin_d = weight_norm(self.lin_d, name='weight', dim=0)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_s.reset_parameters()
        self.lin_d.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        num_nodes = size or x[0].size(0)

        if self.add_self_loops:
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value,
                num_nodes=num_nodes)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             size=size)
        out = self.lin_s(out)

        if self.degree_norml:
            ones_s = x[0].new_ones((edge_index[0].size()))
            ones_d = ones_s if x[1] is None else x[1].new_ones((edge_index[1].size()))
            deg_s = scatter(ones_s, edge_index[0], 
                            dim=0, dim_size=num_nodes, reduce='sum')
            deg_s = deg_s.pow_(-0.5)
            deg_d = scatter(ones_d, edge_index[1],
                            dim=0, dim_size=num_nodes, reduce='sum')
            deg_d = deg_d.pow_(-0.5)
            deg = deg_s * deg_d
            out = out / deg.unsqueeze(-1).clamp(min=1)
    
        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_d(x_r)

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'degree_norml={self.degree_norml}, weight_norml2={self.weight_norml2}, '
                f'bias={self.bias}, add_self_loops={self.add_self_loops})')
