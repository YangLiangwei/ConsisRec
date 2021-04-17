import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConsis(nn.Module):

    def __init__(self, node_enc, r2e):
        super(GraphConsis, self).__init__()
        self.node_enc = node_enc
        self.embed_dim = node_enc.embed_dim
        # self.w1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        # self.w2 = nn.Linear(self.embed_dim, 32)
        # self.w3 = nn.Linear(32, 16)
        # self.w4 = nn.Linear(16, 1)

        # self.r2e = r2e
        # self.bn1 = nn.BatchNorm1d(self.embed_dim)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.bn3 = nn.BatchNorm1d(16)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.node_enc(nodes_u, nodes_v, uv = False)
        embeds_v = self.node_enc(nodes_v, nodes_u, uv = True)
        scores = torch.mm(embeds_u, embeds_v.t()).diagonal()
        return scores

        # x_uv = torch.cat((embeds_u, embeds_v), 1)
        # x = F.relu(self.bn1(self.w1(x_uv)))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.bn2(self.w2(x)))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.bn3(self.w3(x)))
        # x = F.dropout(x, training = self.training)
        # scores = self.w4(x)
        # return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)
