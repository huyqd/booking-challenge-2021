import random

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect  # noqa


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
    LOGITS = []
    TARGETS = []

    with torch.no_grad():
        # with tqdm_logging_redirect():
        for batch in (bar := tqdm(loader)):
            batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}

            out_dict = model(batch)
            logits = out_dict["logits"]
            loss = out_dict["loss"]
            loss_np = loss.detach().cpu().numpy()
            target = batch["target"]
            LOGITS.append(logits.detach())
            TARGETS.append(target.detach())
            val_loss.append(loss_np)

            smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
            bar.set_description(f"loss: {loss_np:.5f}, smooth loss: {smooth_loss:.5f}")

        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    return val_loss, LOGITS, TARGETS


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
    torch.save(checkpoint, f"./checkpoints/{fname}/{fname}_{fold}_{seed}.pt")


def load_checkpoint(
    fold,
    seed,
    device,
    fname,
    NUM_CITIES,
    NUM_HOTELS,
    NUM_DEVICE,
    LOW_CITY,
    lag_cities,
    lag_countries,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    DROPOUT_RATE,
):
    model = Net(
        NUM_CITIES + 1,
        NUM_HOTELS + 1,
        NUM_DEVICE,
        LOW_CITY,
        lag_cities,
        lag_countries,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        loss=False,
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
        self.lag_cities_ = data[lag_cities].values
        self.mn = data["mn"].values - 1
        self.dy1 = data["dy1"].values
        self.dy2 = data["dy2"].values
        self.length = data["length"].values
        self.trip_length = data["trip_length"].values
        self.N = data["N"].values
        self.log_icount = data["log_icount"].values
        self.log_dcount = data["log_dcount"].values
        self.lag_countries_ = data[lag_countries].values
        self.first_city = data["first_city"].values
        self.first_country = data["first_country"].values
        self.booker_country_ = data["booker_country_"].values
        self.device_class_ = data["device_class_"].values
        self.lapse = data["lapse"].values
        self.season = data["season"].values
        self.weekend = data["weekend"].values
        if target is None:
            self.target = None
        else:
            self.target = data[target].values

    def __len__(self):
        return len(self.lag_cities_)

    def __getitem__(self, idx: int):
        input_dict = {
            "lag_cities_": torch.tensor(self.lag_cities_[idx], dtype=torch.long),
            "mn": torch.tensor([self.mn[idx]], dtype=torch.long),
            "dy1": torch.tensor([self.dy1[idx]], dtype=torch.long),
            "dy2": torch.tensor([self.dy2[idx]], dtype=torch.long),
            "length": torch.tensor([self.length[idx]], dtype=torch.float),
            "trip_length": torch.tensor([self.trip_length[idx]], dtype=torch.float),
            "N": torch.tensor([self.N[idx]], dtype=torch.float),
            "log_icount": torch.tensor([self.log_icount[idx]], dtype=torch.float),
            "log_dcount": torch.tensor([self.log_dcount[idx]], dtype=torch.float),
            "lag_countries_": torch.tensor(self.lag_countries_[idx], dtype=torch.long),
            "first_city": torch.tensor([self.first_city[idx]], dtype=torch.long),
            "first_country": torch.tensor([self.first_country[idx]], dtype=torch.long),
            "booker_country_": torch.tensor([self.booker_country_[idx]], dtype=torch.long),
            "device_class_": torch.tensor([self.device_class_[idx]], dtype=torch.long),
            "lapse": torch.tensor([self.lapse[idx]], dtype=torch.float),
            "season": torch.tensor([self.season[idx]], dtype=torch.long),
            "weekend": torch.tensor([self.weekend[idx]], dtype=torch.long),
        }
        if self.target is not None:
            input_dict["target"] = torch.tensor([self.target[idx]], dtype=torch.long)
        return input_dict


class Net(nn.Module):
    def __init__(
        self,
        num_cities,
        num_countries,
        num_devices,
        low_city,
        lag_cities,
        lag_countries,
        embedding_dim,
        hidden_dim,
        dropout_rate,
        loss=True,
    ):
        super(Net, self).__init__()
        self.loss = loss
        self.low_city = low_city
        self.dropout_rate = dropout_rate

        self.cities_embeddings = nn.Embedding(num_cities, embedding_dim)
        self.cities_embeddings.weight.data.normal_(0.0, 0.01)
        print("city embedding data shape", self.cities_embeddings.weight.shape)

        self.countries_embeddings = nn.Embedding(num_countries, embedding_dim)
        self.countries_embeddings.weight.data.normal_(0.0, 0.01)
        print("country embedding data shape", self.countries_embeddings.weight.shape)

        self.mn_embeddings = nn.Embedding(12, embedding_dim)
        self.mn_embeddings.weight.data.normal_(0.0, 0.01)

        self.dy1_embeddings = nn.Embedding(7, embedding_dim)
        self.dy1_embeddings.weight.data.normal_(0.0, 0.01)

        self.dy2_embeddings = nn.Embedding(7, embedding_dim)
        self.dy2_embeddings.weight.data.normal_(0.0, 0.01)

        # self.season_embeddings = nn.Embedding(7, embedding_dim)
        # self.season_embeddings.weight.data.normal_(0., 0.01)

        self.weekend_embeddings = nn.Embedding(2, embedding_dim)
        self.weekend_embeddings.weight.data.normal_(0.0, 0.01)

        self.linear_length = nn.Linear(1, embedding_dim, bias=False)
        self.norm_length = nn.BatchNorm1d(embedding_dim)
        self.activate_length = nn.ReLU()

        self.linear_trip_length = nn.Linear(1, embedding_dim, bias=False)
        self.norm_trip_length = nn.BatchNorm1d(embedding_dim)
        self.activate_trip_length = nn.ReLU()

        self.linear_N = nn.Linear(1, embedding_dim, bias=False)
        self.norm_N = nn.BatchNorm1d(embedding_dim)
        self.activate_N = nn.ReLU()

        self.linear_log_icount = nn.Linear(1, embedding_dim, bias=False)
        self.norm_log_icount = nn.BatchNorm1d(embedding_dim)
        self.activate_log_icount = nn.ReLU()

        self.linear_log_dcount = nn.Linear(1, embedding_dim, bias=False)
        self.norm_log_dcount = nn.BatchNorm1d(embedding_dim)
        self.activate_log_dcount = nn.ReLU()

        self.devices_embeddings = nn.Embedding(num_devices, embedding_dim)
        self.devices_embeddings.weight.data.normal_(0.0, 0.01)
        print("device_embeddings data shape", self.devices_embeddings.weight.shape)

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
        lag_embed = self.get_embed(input_dict["lag_cities_"], self.cities_embeddings)
        lag_countries_embed = self.get_embed(input_dict["lag_countries_"], self.countries_embeddings)
        mn_embed = self.get_embed(input_dict["mn"], self.mn_embeddings)
        dy1_embed = self.get_embed(input_dict["dy1"], self.dy1_embeddings)
        dy2_embed = self.get_embed(input_dict["dy2"], self.dy2_embeddings)
        # season_embed = self.get_embed(input_dict['season'], self.season_embeddings)
        weekend_embed = self.get_embed(input_dict["weekend"], self.weekend_embeddings)
        length = input_dict["length"]
        length_embed = self.activate_length(self.norm_length(self.linear_length(length)))
        trip_length = input_dict["trip_length"]
        trip_length_embed = self.activate_trip_length(self.norm_trip_length(self.linear_trip_length(trip_length)))
        N = input_dict["N"]
        N_embed = self.activate_N(self.norm_N(self.linear_N(N)))
        lapse = input_dict["lapse"]
        lapse_embed = self.activate_lapse(self.norm_lapse(self.linear_lapse(lapse)))
        log_icount = input_dict["log_icount"]
        log_icount_embed = self.activate_log_icount(self.norm_log_icount(self.linear_log_icount(log_icount)))
        log_dcount = input_dict["length"]
        log_dcount_embed = self.activate_log_dcount(self.norm_log_dcount(self.linear_log_dcount(log_dcount)))
        first_city_embed = self.get_embed(input_dict["first_city"], self.cities_embeddings)
        first_country_embed = self.get_embed(input_dict["first_country"], self.countries_embeddings)
        booker_country_embed = self.get_embed(input_dict["booker_country_"], self.countries_embeddings)
        device_embed = self.get_embed(input_dict["device_class_"], self.devices_embeddings)
        x = (
            mn_embed
            + dy1_embed
            + dy2_embed
            + length_embed
            + log_icount_embed
            + log_dcount_embed
            + first_city_embed
            + first_country_embed
            + booker_country_embed
            + device_embed
            + trip_length_embed
            + N_embed
            + lapse_embed
            + weekend_embed
        )
        x = torch.cat([lag_embed, lag_countries_embed, x], -1)
        x = self.activate1(self.norm1(self.linear1(x)))
        x = self.dropout1(x)
        x = x + self.activate2(self.norm2(self.linear2(x)))
        x = self.dropout2(x)
        x = self.activate3(self.norm3(self.linear3(x)))
        x = self.dropout3(x)
        logits = F.linear(x, self.cities_embeddings.weight, bias=self.output_layer_bias)
        output_dict = {"logits": logits}
        if self.loss:
            target = input_dict["target"].squeeze(1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.low_city)
            loss = loss_fct(logits, target)
            output_dict["loss"] = loss
        return output_dict


def seed_torch(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
