import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()

        # Initialize learnable parameters
        self.w1 = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.empty(1, dtype=torch.float32))

        # Initialize weights
        nn.init.normal_(self.w1, mean=0, std=1)
        nn.init.normal_(self.w2, mean=0, std=1)

    def forward(self, x1, x2, x3):
        w1 = torch.sigmoid(self.w1)
        w2 = torch.sigmoid(self.w2)

        result = (x1 + x2 * w1 + x3 * w2) / (1 + w1 + w2)

        return result


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

        self.season_embeddings = nn.Embedding(4, 4)
        self.season_embeddings.weight.data.normal_(0.0, 0.01)

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
            nn.Linear(743, 768),
            nn.BatchNorm1d(768),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.first_stage_fc = nn.Linear(768, num_cities)
        self.second_stage_gc1 = nn.Linear(768, embedding_dim)
        self.second_stage_gc2 = nn.Linear(768, embedding_dim)

        self.weighted_sum = WeightedSum()

        self.output_layer_bias = nn.Parameter(torch.Tensor(num_cities))
        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch):
        lag_cities_embedding = self.cities_embeddings(batch["lag_cities"])
        hotel_country_embedding = self.hotel_country_embedding(batch["country_lag_1"])
        devices_embedding = self.devices_embeddings(batch["device_class"])
        month_embedding = self.month_embeddings(batch["month"])
        checkin_day_embedding = self.checkin_day_embeddings(batch["checkin_day"])
        checkout_day_embedding = self.checkout_day_embeddings(batch["checkout_day"])
        weekend_embedding = self.weekend_embeddings(batch["weekend"])
        stay_length_embedding = self.stay_length_embeddings(batch["stay_length"])
        trip_length_embedding = self.trip_length_embeddings(batch["trip_length"])
        season_embedding = self.season_embeddings(batch["season"])
        first_city_embedding = self.first_city_embeddings(batch["first_city"])
        booker_country_embedding = self.booker_country_embedding(batch["booker_country"])
        lapse_embedding = self.lapse_embeddings(batch["lapse"])
        order_embedding = self.num_order_embeddings(batch["order"])
        inverse_order_embedding = self.num_inverse_order_embeddings(batch["inverse_order"])
        affiliate_id_embedding = self.affiliate_id_embeddings(batch["affiliate_id"])

        lag_cities_embedding = self.gru(lag_cities_embedding)
        cat_embeddings = torch.cat(
            [
                lag_cities_embedding,
                hotel_country_embedding,
                devices_embedding,
                month_embedding,
                checkin_day_embedding,
                checkout_day_embedding,
                weekend_embedding,
                stay_length_embedding,
                trip_length_embedding,
                season_embedding,
                first_city_embedding,
                booker_country_embedding,
                lapse_embedding,
                order_embedding,
                inverse_order_embedding,
                affiliate_id_embedding,
            ],
            dim=1,
        )

        x1 = self.fc1(cat_embeddings)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)

        first_stage_logits = self.first_stage_fc(x2)

        output1 = self.second_stage_gc1(x2)
        output2 = self.second_stage_gc2(x3)

        second_stage_logits1 = F.linear(output1, self.cities_embeddings.weight, bias=self.output_layer_bias)
        second_stage_logits2 = F.linear(output2, self.cities_embeddings.weight, bias=self.output_layer_bias)

        sum_logits = self.weighted_sum(first_stage_logits, second_stage_logits1, second_stage_logits2)
        output_dict = {
            "first_stage_logits": first_stage_logits,
            "second_stage_logits1": second_stage_logits1,
            "second_stage_logits2": second_stage_logits2,
            "sum_logits": sum_logits,
        }

        if "target" in batch.keys():
            target = batch["target"].squeeze(1)
            first_stage_loss = self.loss_fct(first_stage_logits, target)
            second_stage_loss1 = self.loss_fct(second_stage_logits1, target)
            second_stage_loss2 = self.loss_fct(second_stage_logits2, target)
            sum_loss = self.loss_fct(sum_logits, target)
            total_loss = first_stage_loss + second_stage_loss1 + second_stage_loss2 + sum_loss
            loss_dict = {
                "first_stage_loss": first_stage_loss,
                "second_stage_loss1": second_stage_loss1,
                "second_stage_loss2": second_stage_loss2,
                "sum_loss": sum_loss,
                "total_loss": total_loss,
            }
            output_dict.update(loss_dict)

        return output_dict
