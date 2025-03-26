import torch.nn as nn
import torch
from torch.nn import functional as F
from torch_scatter import scatter_softmax, scatter_sum

class NHGNN(nn.Module):
    def __init__(self, hidden_dim = 128, num_heads = 2, num_node_type = 13):
        super(NHGNN, self).__init__()
        self.num_node_type = num_node_type
        self.node_MLP = nn.ModuleDict({
            str(i): nn.Linear(hidden_dim, hidden_dim) for i in range(self.num_node_type)
        })
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.score_MLP = nn.Sequential(
            nn.Linear(hidden_dim, num_heads, bias = False)
        )
        self.edge_MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x, edge_index, edge_attr, node_type_index):
        row, col = edge_index
        n_heads = self.num_heads
        d = int(self.hidden_dim / n_heads)
        for node_type in range(self.num_node_type):
            mask = (node_type_index == node_type)
            if mask.any():
                x[mask] = self.node_MLP[str(node_type)](x[mask])
        x_ = x
        x_emb = x_.view(-1, n_heads, d)
        score_input = self.score_MLP(x_[row] + x_[col] + edge_attr).view(-1, n_heads, 1)
        score = F.leaky_relu(score_input, 0.2)
        alpha = scatter_softmax(src = score, index = row, dim = 0)
        x = scatter_sum(src = alpha * x_emb[col], index = row, dim = 0).view(x.shape[0], -1)
        edge_attr = self.edge_MLP(x[row] + x[col] + edge_attr)
        return x, edge_attr


class HHGNN(nn.Module):
    def __init__(self, hidden_dim = 128, num_node_type = 13):
        super(HHGNN, self).__init__()
        self.num_node_type = num_node_type
        self.node_linear_dict = nn.ModuleDict({ str(i): nn.Linear(hidden_dim, hidden_dim) for i in range(self.num_node_type)})
        self.node_norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
        )
    def forward(self, adj, embeds, node_type_index, node_mask):
        for node_type in range(self.num_node_type):
            mask = (node_type_index == node_type)
            if mask.any():
                embeds[node_mask][mask] = self.node_linear_dict[str(node_type)](embeds[node_mask][mask])
        embeds = self.node_norm(embeds)
        adj_t = adj.permute(0,2,1)
        lat = torch.bmm(adj_t, embeds)
        ret = torch.bmm(adj, lat)
        return ret
