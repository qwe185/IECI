from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor
import torch_geometric.transforms as TOsparese

def group(xs: List[Tensor], beta_1, beta_2) -> Optional[Tensor]:

    if len(xs) == 0:
        return None
    else:
        out = torch.stack(xs)
        if out.numel() == 0:
            return out.view(0, out.size(-1))
        n, d = out.shape[1:]
        final_out = torch.zeros(n, d).cuda()
        mask = torch.all(out[1] == 0, dim=1).cuda()
        final_out[mask] = out[0, mask]
        final_out[~mask] = out[0, ~mask] * beta_1 + out[1, ~mask] * beta_2
        return final_out


class CGEConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        beta_intra=0.7,
        beta_inter=0.3,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout

        # 为句内和句间边的权重
        self.beta_intra = beta_intra
        self.beta_inter = beta_inter

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            # 不同的边类型使用不同的参数
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))

        self.reset_parameters()

    def reset_parameters(self):
        """重置模型参数。"""
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)

    def forward(
        self, x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]) -> Dict[NodeType, Optional[Tensor]]:

        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # 遍历节点类型：
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # 遍历边类型：
        for edge_type, edge_index in edge_index_dict.items():
            edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], value=None,
                                      sparse_sizes=(x.shape[0], x.shape[0]))
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            alpha_src = (x_src * lin_src).sum(dim=-1)

            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            out = self.propagate(edge_index, x=(x_src, x_dst),
                                 alpha=(alpha_src, alpha_dst), size=None)

            out = F.relu(out)
            out_dict[dst_type].append(out)

        # 再次遍历节点类型：
        for node_type, outs in out_dict.items():
            if len(outs) == 1:
                out_dict[node_type] = outs[0]
            elif len(outs) == 0:
                out_dict[node_type] = None
                continue
            else:
                out = group(outs, self.beta_intra, self.beta_inter)
                out_dict[node_type] = out

        return out_dict

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        """返回模型的字符串表示形式。"""
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')
