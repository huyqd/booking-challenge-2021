import torch
from torch import nn as nn
from torch.nn import functional as F


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
