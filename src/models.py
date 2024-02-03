import torch
import torch.nn.functional as F
from torch import nn as nn


class Linear(nn.Module):
    def __init__(self, H, activation="relu"):
        super(Linear, self).__init__()
        self.dense = nn.Linear(H, H)
        self.bn = nn.BatchNorm1d(H)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Activation not supported")

    def forward(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x = self.activation(x)
        return x


class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()
        self.w1 = nn.Parameter(torch.Tensor(1))  # Initialize with appropriate shape
        self.w2 = nn.Parameter(torch.Tensor(1))  # Initialize with appropriate shape
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x1, x2, x3):
        w1 = torch.sigmoid(self.w1)
        w2 = torch.sigmoid(self.w2)
        x1 = x1.detach()
        x2 = x2.detach()
        x3 = x3.detach()
        return (x1 + x2 * w1 + x3 * w2) / (1 + w1 + w2)


class EmbDotSoftMax(nn.Module):
    def __init__(self, EC):
        super(EmbDotSoftMax, self).__init__()
        self.d1 = nn.Linear(EC, EC)

    def forward(self, x, top_city_emb, top_city_id, prob):
        emb_pred = self.d1(x)  # B, EC
        emb_pred = emb_pred.unsqueeze(1)  # B, 1, EC
        x = emb_pred * top_city_emb  # B, N_CITY, EC
        x = torch.sum(x, dim=2)  # B, N_CITY
        x = F.softmax(x, dim=1)  # B, N_CITY

        rowids = torch.arange(0, x.shape[0])  # B
        rowids = rowids.unsqueeze(1).repeat(1, N_CITY)  # B, N_CITY

        idx = torch.stack([rowids, top_city_id], dim=2)  # B, N_CITY, 2
        idx = idx.to(torch.int32)
        prob = torch.scatter_add(torch.zeros_like(prob), dim=0, index=idx, src=x) + 1e-6
        return prob


class YourModel(nn.Module):
    def __init__(self, emb_map, FEATURES, N_CITY, EC, t_ct):
        super(YourModel, self).__init__()

        # Assuming emb_map is a dictionary mapping feature names to embedding dimensions
        self.FEATURES = FEATURES
        self.N_CITY = N_CITY
        self.embedding_layers = nn.ModuleList()
        for k, f in enumerate(FEATURES):
            i, j = emb_map[f]
            if f.startswith("city_id"):
                e_city = nn.Embedding(i, j)
                self.embedding_layers.append(e_city)
            else:
                e = nn.Embedding(i, j)
                self.embedding_layers.append(e)

        self.gru = nn.GRU(input_size=EC, hidden_size=EC, batch_first=True)
        self.linear_concat = nn.Linear(sum([emb_map[f][1] for f in FEATURES]) + EC, 512 + 256)
        self.linear_x1 = Linear(512 + 256, "relu")
        self.linear_x2 = Linear(512 + 256, "relu")

        self.prob_layer = nn.Linear(512 + 256, t_ct)
        self.emb_dot_softmax_1 = EmbDotSoftMax(EC)
        self.emb_dot_softmax_2 = EmbDotSoftMax(EC)
        self.weighted_sum = WeightedSum()

    def forward(self, x):
        embs = []
        city_embs = []

        for k, f in enumerate(self.FEATURES):
            if f.startswith("city_id"):
                city_embs.append(self.embedding_layers[k](x[:, k]))
            else:
                embs.append(self.embedding_layers[k](x[:, k]))

        xc = torch.stack(city_embs, dim=1)  # B, T, F
        xc, _ = self.gru(xc)

        x = torch.cat(embs, dim=1)
        x = torch.cat([x, xc[:, -1, :]], dim=1)

        x1 = self.linear_concat(x)
        x2 = self.linear_x1(x1)

        prob = self.prob_layer(x2)
        _, top_city_id = torch.topk(prob, self.N_CITY, dim=1)
        top_city_emb = torch.stack([self.embedding_layers[0](top_city_id[:, i]) for i in range(self.N_CITY)], dim=1)

        x1 = self.linear_x2(x1)
        prob_1 = self.emb_dot_softmax_1(x1, top_city_emb, top_city_id, prob)
        prob_2 = self.emb_dot_softmax_2(x2, top_city_emb, top_city_id, prob)

        prob_ws = self.weighted_sum(prob, prob_1, prob_2)

        return prob, prob_1, prob_2, prob_ws
