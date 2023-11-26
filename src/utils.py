import random

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect  # noqa
from pathlib import Path


def train_epoch(loader, model, optimizer, scheduler, device):
    model.train()
    train_loss = []

    # with tqdm_logging_redirect():
    for batch in (bar := tqdm(loader)):
        optimizer.zero_grad()

        batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}
        out_dict = model(batch)
        loss = out_dict["loss"]
        loss_np = loss.detach().cpu().numpy()

        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)

        bar.set_description(f"loss: {loss_np:.5f}, smooth loss: {smooth_loss:.5f}")

    return train_loss


def val_epoch(loader, model, device):
    model.eval()
    val_loss = []
    logits = []
    targets = []

    with torch.no_grad():
        # with tqdm_logging_redirect():
        for batch in (bar := tqdm(loader)):
            batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}

            out_dict = model(batch)
            batch_logits = out_dict["logits"]
            loss = out_dict["loss"]
            loss_np = loss.detach().cpu().numpy()
            target = batch["target"]
            logits.append(batch_logits.detach())
            targets.append(target.detach())
            val_loss.append(loss_np)

            smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
            bar.set_description(f"loss: {loss_np:.5f}, smooth loss: {smooth_loss:.5f}")

        val_loss = np.mean(val_loss)

    logits = torch.cat(logits).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    return val_loss, logits, targets


def topk(preds, target, k=4):
    topk_ = preds.argsort(axis=1)[:, ::-1][:, :k]
    acc = np.max(topk_ == target, axis=1)
    acc = np.mean(acc)

    return acc


def save_checkpoint(model, optimizer, scheduler, epoch, best_score, fold, seed, fname):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    checkpoint_path = Path(f"./checkpoints/{fname}")
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path / f"{fname}_{fold}_{seed}.pt")


def load_checkpoint(
    fold,
    seed,
    device,
    fname,
    num_cities,
    num_countries,
    num_devices,
    low_frequency_city_index,
    lag_cities,
    lag_countries,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    DROPOUT_RATE,
):
    model = MLP(
        num_cities + 1,
        num_countries + 1,
        num_devices,
        low_frequency_city_index,
        lag_cities,
        lag_countries,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    checkpoint = torch.load(f"./checkpoints/{fname}/{fname}_{fold}_{seed}.pt")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


class BookingDataset(Dataset):
    def __init__(
        self,
        data,
        lag_cities,
        lag_countries,
        target=None,
    ):
        super(BookingDataset, self).__init__()
        self.lag_cities = data[lag_cities].values
        self.month = data["month"].values - 1
        self.checkin_day = data["checkin_day"].values
        self.checkout_day = data["checkout_day"].values
        self.stay_length = data["stay_length"].values
        self.trip_length = data["trip_length"].values
        self.num_visited = data["num_visited"].values
        self.log_inverse_order = data["log_inverse_order"].values
        self.log_order = data["log_order"].values
        self.lag_countries = data[lag_countries].values
        self.first_city = data["first_city"].values
        self.first_country = data["first_country"].values
        self.booker_country = data["booker_country"].values
        self.device_class = data["device_class"].values
        self.lapse = data["lapse"].values
        self.season = data["season"].values
        self.weekend = data["weekend"].values
        if target is None:
            self.target = None
        else:
            self.target = data[target].values

    def __len__(self):
        return len(self.lag_cities)

    def __getitem__(self, idx: int):
        input_dict = {
            "lag_cities": torch.tensor(self.lag_cities[idx], dtype=torch.long),
            "month": torch.tensor([self.month[idx]], dtype=torch.long),
            "checkin_day": torch.tensor([self.checkin_day[idx]], dtype=torch.long),
            "checkout_day": torch.tensor([self.checkout_day[idx]], dtype=torch.long),
            "stay_length": torch.tensor([self.stay_length[idx]], dtype=torch.float),
            "trip_length": torch.tensor([self.trip_length[idx]], dtype=torch.float),
            "num_visited": torch.tensor([self.num_visited[idx]], dtype=torch.float),
            "log_inverse_order": torch.tensor([self.log_inverse_order[idx]], dtype=torch.float),
            "log_order": torch.tensor([self.log_order[idx]], dtype=torch.float),
            "lag_countries": torch.tensor(self.lag_countries[idx], dtype=torch.long),
            "first_city": torch.tensor([self.first_city[idx]], dtype=torch.long),
            "first_country": torch.tensor([self.first_country[idx]], dtype=torch.long),
            "booker_country": torch.tensor([self.booker_country[idx]], dtype=torch.long),
            "device_class": torch.tensor([self.device_class[idx]], dtype=torch.long),
            "lapse": torch.tensor([self.lapse[idx]], dtype=torch.float),
            "season": torch.tensor([self.season[idx]], dtype=torch.long),
            "weekend": torch.tensor([self.weekend[idx]], dtype=torch.long),
        }
        if self.target is not None:
            input_dict["target"] = torch.tensor([self.target[idx]], dtype=torch.long)
        return input_dict


class MLP(nn.Module):
    def __init__(
        self,
        num_cities,
        num_countries,
        num_devices,
        low_frequency_city_index,
        lag_cities,
        lag_countries,
        embedding_dim,
        hidden_dim,
        dropout_rate,
    ):
        super(MLP, self).__init__()
        self.low_frequency_city_index = low_frequency_city_index
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.low_frequency_city_index)
        self.dropout_rate = dropout_rate

        self.cities_embeddings = nn.Embedding(num_cities, embedding_dim)
        self.cities_embeddings.weight.data.normal_(0.0, 0.01)
        print("city embedding data shape", self.cities_embeddings.weight.shape)

        self.countries_embeddings = nn.Embedding(num_countries, embedding_dim)
        self.countries_embeddings.weight.data.normal_(0.0, 0.01)
        print("country embedding data shape", self.countries_embeddings.weight.shape)

        self.devices_embeddings = nn.Embedding(num_devices, embedding_dim)
        self.devices_embeddings.weight.data.normal_(0.0, 0.01)
        print("device_embeddings data shape", self.devices_embeddings.weight.shape)

        self.month_embeddings = nn.Embedding(12, embedding_dim)
        self.month_embeddings.weight.data.normal_(0.0, 0.01)

        self.checkin_day_embeddings = nn.Embedding(7, embedding_dim)
        self.checkin_day_embeddings.weight.data.normal_(0.0, 0.01)

        self.checkout_day_embeddings = nn.Embedding(7, embedding_dim)
        self.checkout_day_embeddings.weight.data.normal_(0.0, 0.01)

        # self.season_embeddings = nn.Embedding(7, embedding_dim)
        # self.season_embeddings.weight.data.normal_(0., 0.01)

        self.weekend_embeddings = nn.Embedding(2, embedding_dim)
        self.weekend_embeddings.weight.data.normal_(0.0, 0.01)

        self.linear_stay_length = nn.Linear(1, embedding_dim, bias=False)
        self.norm_stay_length = nn.BatchNorm1d(embedding_dim)
        self.activate_stay_length = nn.ReLU()

        self.linear_trip_length = nn.Linear(1, embedding_dim, bias=False)
        self.norm_trip_length = nn.BatchNorm1d(embedding_dim)
        self.activate_trip_length = nn.ReLU()

        self.linear_num_visited = nn.Linear(1, embedding_dim, bias=False)
        self.norm_num_visited = nn.BatchNorm1d(embedding_dim)
        self.activate_num_visited = nn.ReLU()

        self.linear_log_inverse_order = nn.Linear(1, embedding_dim, bias=False)
        self.norm_log_inverse_order = nn.BatchNorm1d(embedding_dim)
        self.activate_log_inverse_order = nn.ReLU()

        self.linear_log_order = nn.Linear(1, embedding_dim, bias=False)
        self.norm_log_order = nn.BatchNorm1d(embedding_dim)
        self.activate_log_order = nn.ReLU()

        self.linear_lapse = nn.Linear(1, embedding_dim, bias=False)
        self.norm_lapse = nn.BatchNorm1d(embedding_dim)
        self.activate_lapse = nn.ReLU()

        self.linear1 = nn.Linear((len(lag_cities) + len(lag_countries) + 1) * embedding_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.activate1 = nn.PReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.activate2 = nn.PReLU()
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.linear3 = nn.Linear(hidden_dim, embedding_dim)
        self.norm3 = nn.BatchNorm1d(embedding_dim)
        self.activate3 = nn.PReLU()
        self.dropout3 = nn.Dropout(self.dropout_rate)
        self.output_layer_bias = nn.Parameter(
            torch.Tensor(
                num_cities,
            )
        )
        self.output_layer_bias.data.normal_(0.0, 0.01)

    def get_embed(self, x, embed):
        bs = x.shape[0]
        x = embed(x)
        # lag_embed.shape: bs, x.shape[1], embedding_dim
        x = x.view(bs, -1)
        return x

    def forward(self, input_dict):
        lag_cities_embeddings = self.get_embed(input_dict["lag_cities"], self.cities_embeddings)
        lag_countries_embeddings = self.get_embed(input_dict["lag_countries"], self.countries_embeddings)
        month_embeddings = self.get_embed(input_dict["month"], self.month_embeddings)
        checkin_day_embeddings = self.get_embed(input_dict["checkin_day"], self.checkin_day_embeddings)
        checkout_day_embeddings = self.get_embed(input_dict["checkout_day"], self.checkout_day_embeddings)
        # season_embed = self.get_embed(input_dict['season'], self.season_embeddings)
        weekend_embeddings = self.get_embed(input_dict["weekend"], self.weekend_embeddings)
        stay_length = input_dict["stay_length"]
        stay_length_embeddings = self.activate_stay_length(self.norm_stay_length(self.linear_stay_length(stay_length)))
        trip_length = input_dict["trip_length"]
        trip_length_embeddings = self.activate_trip_length(self.norm_trip_length(self.linear_trip_length(trip_length)))
        num_visited = input_dict["num_visited"]
        num_visited_embeddings = self.activate_num_visited(self.norm_num_visited(self.linear_num_visited(num_visited)))
        lapse = input_dict["lapse"]
        lapse_embeddings = self.activate_lapse(self.norm_lapse(self.linear_lapse(lapse)))
        log_inverse_order = input_dict["log_inverse_order"]
        log_inverse_order_embeddings = self.activate_log_inverse_order(
            self.norm_log_inverse_order(self.linear_log_inverse_order(log_inverse_order))
        )
        log_order = input_dict["log_order"]
        log_order_embeddings = self.activate_log_order(self.norm_log_order(self.linear_log_order(log_order)))
        first_city_embeddings = self.get_embed(input_dict["first_city"], self.cities_embeddings)
        first_country_embeddings = self.get_embed(input_dict["first_country"], self.countries_embeddings)
        booker_country_embeddings = self.get_embed(input_dict["booker_country"], self.countries_embeddings)
        device_embeddings = self.get_embed(input_dict["device_class"], self.devices_embeddings)
        x = (
            month_embeddings
            + checkin_day_embeddings
            + checkout_day_embeddings
            + stay_length_embeddings
            + log_inverse_order_embeddings
            + log_order_embeddings
            + first_city_embeddings
            + first_country_embeddings
            + booker_country_embeddings
            + device_embeddings
            + trip_length_embeddings
            + num_visited_embeddings
            + lapse_embeddings
            + weekend_embeddings
        )
        x = torch.cat([lag_cities_embeddings, lag_countries_embeddings, x], -1)
        x = self.activate1(self.norm1(self.linear1(x)))
        x = self.dropout1(x)
        x = x + self.activate2(self.norm2(self.linear2(x)))
        x = self.dropout2(x)
        x = self.activate3(self.norm3(self.linear3(x)))
        x = self.dropout3(x)
        logits = F.linear(x, self.cities_embeddings.weight, bias=self.output_layer_bias)
        output_dict = {"logits": logits}

        if "target" in input_dict.keys():
            target = input_dict["target"].squeeze(1)
            loss = self.loss_fct(logits, target)
            output_dict["loss"] = loss

        return output_dict


def seed_torch(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
