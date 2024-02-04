import torch
from torch import nn as nn
from torch.nn import functional as F


class MSGRUSMF(nn.Module):
    def __init__(
        self,
        num_cities,
        num_hotel_countries,
        num_booker_countries,
        num_devices,
        trip_length,
        stay_length,
        lapse,
        num_affiliates,
        num_order,
        num_inverse_order,
        loss_ignore_index,
        embedding_dim,
        num_lags,
        dropout,
    ):
        super().__init__()
        self.cities_embeddings = nn.Embedding(num_cities, 400)
        self.cities_embeddings.weight.data.normal_(0.0, 0.01)

        self.hotel_country_embedding = nn.Embedding(num_hotel_countries, 14)
        self.hotel_country_embedding.weight.data.normal_(0.0, 0.01)

        self.booker_country_embedding = nn.Embedding(num_booker_countries, 3)
        self.booker_country_embedding.weight.data.normal_(0.0, 0.01)

        self.devices_embeddings = nn.Embedding(num_devices, 2)
        self.devices_embeddings.weight.data.normal_(0.0, 0.01)

        self.first_city_embeddings = nn.Embedding(num_cities, 200)
        self.first_city_embeddings.weight.data.normal_(0.0, 0.01)

        self.affiliate_id_embeddings = nn.Embedding(num_affiliates, 60)
        self.affiliate_id_embeddings.weight.data.normal_(0.0, 0.01)

        self.month_embeddings = nn.Embedding(12, 4)
        self.month_embeddings.weight.data.normal_(0.0, 0.01)

        self.checkin_day_embeddings = nn.Embedding(7, 3)
        self.checkin_day_embeddings.weight.data.normal_(0.0, 0.01)

        self.checkout_day_embeddings = nn.Embedding(7, 3)
        self.checkout_day_embeddings.weight.data.normal_(0.0, 0.01)

        self.weekend_embeddings = nn.Embedding(2, 3)
        self.weekend_embeddings.weight.data.normal_(0.0, 0.01)

        self.trip_length_embeddings = nn.Embedding(trip_length, 18)
        self.trip_length_embeddings.weight.data.normal_(0.0, 0.01)

        self.stay_length_embeddings = nn.Embedding(stay_length, 6)
        self.stay_length_embeddings.weight.data.normal_(0.0, 0.01)

        self.lapse_embeddings = nn.Embedding(lapse, 9)
        self.lapse_embeddings.weight.data.normal_(0.0, 0.01)

        self.num_order_embeddings = nn.Embedding(num_order, 7)
        self.num_order_embeddings.weight.data.normal_(0.0, 0.01)

        self.num_inverse_order_embeddings = nn.Embedding(num_inverse_order, 7)
        self.num_inverse_order_embeddings.weight.data.normal_(0.0, 0.01)

        self.gru = nn.GRU(
            input_size=num_lags * embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(739, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_layer_bias = nn.Parameter(torch.Tensor(num_cities))
        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=loss_ignore_index)

    def forward(self, batch):
        lag_cities_embedding = self.cities_embeddings(batch["lag_cities"])
        lag_countries_embedding = self.hotel_country_embedding(batch["lag_countries"])
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
        first_country_embedding = self.hotel_country_embedding(batch["first_country"])
        booker_country_embedding = self.hotel_country_embedding(batch["booker_country"])

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
