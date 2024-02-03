from torch import nn as nn
import torch


class MLPSMF(nn.Module):
    def __init__(
        self,
        num_cities,
        num_countries,
        num_devices,
        num_lags,
        loss_ignore_index,
        embedding_dim,
        hidden_dim,
        dropout,
    ):
        super().__init__()
        self.cities_embeddings = nn.Embedding(num_cities, embedding_dim)
        self.cities_embeddings.weight.data.normal_(0.0, 0.01)

        self.countries_embeddings = nn.Embedding(num_countries, embedding_dim)
        self.countries_embeddings.weight.data.normal_(0.0, 0.01)

        self.devices_embeddings = nn.Embedding(num_devices, embedding_dim)
        self.devices_embeddings.weight.data.normal_(0.0, 0.01)

        self.month_embeddings = nn.Embedding(12, embedding_dim)
        self.month_embeddings.weight.data.normal_(0.0, 0.01)

        self.checkin_day_embeddings = nn.Embedding(7, embedding_dim)
        self.checkin_day_embeddings.weight.data.normal_(0.0, 0.01)

        self.checkout_day_embeddings = nn.Embedding(7, embedding_dim)
        self.checkout_day_embeddings.weight.data.normal_(0.0, 0.01)

        self.weekend_embeddings = nn.Embedding(2, embedding_dim)
        self.weekend_embeddings.weight.data.normal_(0.0, 0.01)

        self.stay_length_embeddings = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, embedding_dim)),
        )

        self.trip_length_embeddings = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, embedding_dim)),
        )

        self.num_visited_embeddings = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, embedding_dim)),
        )

        self.log_order_embeddings = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, embedding_dim)),
        )

        self.log_inverse_order_embeddings = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, embedding_dim)),
        )

        self.lapse_embeddings = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, embedding_dim)),
        )
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * (num_lags * 2 + 1), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.output_layer_bias = nn.Parameter(torch.Tensor(num_cities))
        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=loss_ignore_index)

    def forward(self, batch):
        lag_cities_embedding = self.cities_embeddings(batch["lag_cities"])
        lag_countries_embedding = self.countries_embeddings(batch["lag_countries"])
        devices_embedding = self.devices_embeddings(batch["device_class"])
        month_embedding = self.month_embeddings(batch["month"])
        checkin_day_embedding = self.checkin_day_embeddings(batch["checkin_day"])
        checkout_day_embedding = self.checkout_day_embeddings(batch["checkout_day"])
        weekend_embedding = self.weekend_embeddings(batch["weekend"])
        stay_length_embedding = self.stay_length_embeddings(batch["stay_length"])
        trip_length_embedding = self.trip_length_embeddings(batch["trip_length"])
        num_visited_embedding = self.num_visited_embeddings(batch["num_visited"])
        log_order_embedding = self.log_order_embeddings(batch["log_order"])
        log_inverse_order_embedding = self.log_inverse_order_embeddings(batch["log_inverse_order"])
        lapse_embedding = self.lapse_embeddings(batch["lapse"])
        first_city_embedding = self.cities_embeddings(batch["first_city"])
        first_country_embedding = self.countries_embeddings(batch["first_country"])
        booker_country_embedding = self.countries_embeddings(batch["booker_country"])

        sum_embeddings = torch.sum(
            torch.cat(
                [
                    devices_embedding,
                    month_embedding,
                    checkin_day_embedding,
                    checkout_day_embedding,
                    weekend_embedding,
                    stay_length_embedding,
                    trip_length_embedding,
                    num_visited_embedding,
                    log_order_embedding,
                    log_inverse_order_embedding,
                    lapse_embedding,
                    first_city_embedding,
                    first_country_embedding,
                    booker_country_embedding,
                ],
                dim=1,
            ),
            keepdim=True,
            dim=1,
        )

        output = torch.cat([lag_cities_embedding, lag_countries_embedding, sum_embeddings], dim=1)
        bs = output.shape[0]
        output = output.view(bs, -1)
        output = self.fc(output)
        output = F.linear(output, self.cities_embeddings.weight, bias=self.output_layer_bias)

        output_dict = {"logits": output}

        if "target" in batch.keys():
            target = batch["target"].squeeze(1)
            loss = self.loss_fct(output, target)
            output_dict["loss"] = loss

        return output_dict


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
    def __init__(self, emb_map, FEATURES, EC, t_ct):
        super(YourModel, self).__init__()

        # Assuming emb_map is a dictionary mapping feature names to embedding dimensions
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

        for k, f in enumerate(FEATURES):
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
        _, top_city_id = torch.topk(prob, N_CITY, dim=1)
        top_city_emb = torch.stack([self.embedding_layers[0](top_city_id[:, i]) for i in range(N_CITY)], dim=1)

        x1 = self.linear_x2(x1)
        prob_1 = self.emb_dot_softmax_1(x1, top_city_emb, top_city_id, prob)
        prob_2 = self.emb_dot_softmax_2(x2, top_city_emb, top_city_id, prob)

        prob_ws = self.weighted_sum(prob, prob_1, prob_2)

        return prob, prob_1, prob_2, prob_ws
