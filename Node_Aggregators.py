import torch
import torch.nn as nn

class Node_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, relation_token, cuda="cpu"):
        super(Node_Aggregator, self).__init__()
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.relation_token = relation_token
        self.device = cuda
        self.embed_dim = embed_dim
        self.relation_att = torch.randn(2 * embed_dim, requires_grad=True).to(self.device)
        self.linear = nn.Linear(2 * embed_dim, embed_dim)
        self.softmax1 = nn.Softmax(dim = 0)
        self.softmax2 = nn.Softmax(dim = 0)

    def neighbor_agg(self, query, history_feature, relation_feature, percent):
        prob = -torch.norm(query - history_feature, dim = 1)
        prob = self.softmax1(prob)
        neighbor_selected = torch.multinomial(prob, max(1,int(percent * len(history_feature))))
        relation_selected = relation_feature[neighbor_selected]
        neighbor_selected = history_feature[neighbor_selected]
        selected = torch.cat((neighbor_selected, relation_selected), 1)
        selected = torch.mm(selected, self.relation_att.unsqueeze(0).t()).squeeze(-1)
        prob = self.softmax2(selected)
        return torch.mm(neighbor_selected.transpose(0,1), prob.unsqueeze(-1)).squeeze(-1)

    def forward(self, self_feats, target_feats, history_uv, history_r, adj, uv, percent):

        embed_matrix = torch.zeros(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)
        query = self.linear(torch.cat((self_feats, target_feats), dim = -1))
        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]

            if uv is True:
                e_uv = self.u2e.weight[history]
                e_neighbor = self.v2e.weight[adj[i]]
                e_uv = torch.cat((e_uv, e_neighbor), 0)
                tmp_label += [self.relation_token] * len(adj[i])
                num_histroy_item += len(adj[i])

            else:
                e_uv = self.v2e.weight[history]
                e_neighbor = self.u2e.weight[adj[i]]
                e_uv = torch.cat((e_uv, e_neighbor), 0)
                tmp_label += [self.relation_token] * len(adj[i])
                num_histroy_item += len(adj[i])

            e_r = self.r2e.weight[tmp_label]
            if num_histroy_item != 0:
                agg = self.neighbor_agg(query[i], e_uv, e_r, percent)
                embed_matrix[i] = agg

        to_feats = embed_matrix
        return to_feats
