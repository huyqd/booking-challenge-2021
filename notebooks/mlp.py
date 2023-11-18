import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from metaflow import FlowSpec, step


def train_epoch(loader, model, optimizer, scheduler, device):
    model.train()
    model.zero_grad()
    train_loss = []
    bar = tqdm(range(len(loader)))
    load_iter = iter(loader)
    batch = next(load_iter)
    batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}

    for i in bar:
        old_batch = batch
        if i + 1 < len(loader):
            batch = next(load_iter)
            batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}

        out_dict = model(old_batch)
        logits = out_dict["logits"]
        loss = out_dict["loss"]
        loss_np = loss.detach().cpu().numpy()

        loss.backward()

        optimizer.step()
        scheduler.step()
        for p in model.parameters():
            p.grad = None

        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description("loss: %.5f, smth: %.5f" % (loss_np, smooth_loss))
    return train_loss


def get_top4(preds):
    TOP4 = np.empty((preds.shape[0], 4))
    for i in range(4):
        x = np.argmax(preds, axis=1)
        TOP4[:, i] = x
        x = np.expand_dims(x, axis=1)
        np.put_along_axis(preds, x, -1e10, axis=1)
    return TOP4


def top4(preds, target):
    TOP4 = get_top4(preds)
    acc = np.max(TOP4 == target, axis=1)
    acc = np.mean(acc)
    return acc


def val_epoch(loader, model, device):
    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []

    with torch.no_grad():
        bar = tqdm(range(len(loader)))
        load_iter = iter(loader)
        batch = next(load_iter)
        batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}

        for i in bar:
            old_batch = batch
            if i + 1 < len(loader):
                batch = next(load_iter)
                batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}

            out_dict = model(old_batch)
            logits = out_dict["logits"]
            loss = out_dict["loss"]
            loss_np = loss.detach().cpu().numpy()
            target = old_batch["target"]
            LOGITS.append(logits.detach())
            TARGETS.append(target.detach())
            val_loss.append(loss_np)

            smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
            bar.set_description("loss: %.5f, smth: %.5f" % (loss_np, smooth_loss))

        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    return val_loss, LOGITS, TARGETS


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


class MLPTrain(FlowSpec):
    @step
    def start(self):
        self.next(self.train)

    @step
    def train(self):
        checkpoint_path = Path("./checkpoints") / "mlp"
        input_path = Path("../data/")
        fname = "mlp"

        LOW_CITY_THR = 9
        LAGS = 5
        # Read CSV using Pandas
        raw = pd.read_csv(input_path / "train_and_test_2.csv")
        # Replace 0s in 'city_id' with NaN
        raw.loc[raw["city_id"] == 0, "city_id"] = np.NaN
        # Group by 'city_id' and count 'utrip_id'
        df = raw[(raw.istest == 0) | (raw.icount > 0)].groupby("city_id")["utrip_id"].count().reset_index()
        df.columns = ["city_id", "city_count"]
        raw = raw.merge(df, how="left", on="city_id")
        raw.loc[raw.city_count <= LOW_CITY_THR, "city_id"] = -1
        raw = raw.sort_values(["utrip_id", "checkin"])
        # Factorize categorical columns
        CATS = ["city_id", "hotel_country", "booker_country", "device_class"]
        MAPS = []
        for c in CATS:
            raw[c + "_"], mp = raw[c].factorize()
            MAPS.append(mp)
            print("created", c + "_")
        # Find the index of the "low city" (-1)
        LOW_CITY = np.where(MAPS[0] == -1)[0][0]
        # Number of unique categories for one-hot encoding
        NUM_CITIES = raw.city_id_.max() + 1
        NUM_HOTELS = raw.hotel_country_.max() + 1
        NUM_DEVICE = raw.device_class_.max() + 1
        # Reverse the data for training set
        raw["reverse"] = 0
        rev_raw = raw[raw.istest == 0].copy()
        rev_raw["reverse"] = 1
        rev_raw["utrip_id"] = rev_raw["utrip_id"] + "_r"
        tmp = rev_raw["icount"].values.copy()
        rev_raw["icount"] = rev_raw["dcount"]
        rev_raw["dcount"] = tmp
        rev_raw = rev_raw.sort_values(["utrip_id", "dcount"]).reset_index(drop=True)
        raw = pd.concat([raw, rev_raw]).reset_index(drop=True)
        # Add 'sorting' column
        raw["sorting"] = np.arange(raw.shape[0])

        # Factorize 'utrip_id'
        raw["utrip_id" + "_"], mp = raw["utrip_id"].factorize()

        # Engineer lag features
        lag_cities = []
        lag_countries = []
        for i in range(1, LAGS + 1):
            raw[f"city_id_lag{i}"] = raw.groupby("utrip_id_")["city_id_"].shift(i, fill_value=NUM_CITIES)
            lag_cities.append(f"city_id_lag{i}")
            raw[f"country_lag{i}"] = raw.groupby("utrip_id_")["hotel_country_"].shift(i, fill_value=NUM_CITIES)
            lag_countries.append(f"country_lag{i}")
        # Extract the first city and country for each trip
        tmpD = raw[raw["dcount"] == 0][["utrip_id", "city_id_"]]
        tmpD.columns = ["utrip_id", "first_city"]
        raw = raw.merge(tmpD, on="utrip_id", how="left")
        tmpD = raw[raw["dcount"] == 0][["utrip_id", "hotel_country_"]]
        tmpD.columns = ["utrip_id", "first_country"]
        raw = raw.merge(tmpD, on="utrip_id", how="left")
        # Convert 'checkin' and 'checkout' columns to datetime in Pandas
        raw["checkin"] = pd.to_datetime(raw["checkin"], format="%Y-%m-%d")
        raw["checkout"] = pd.to_datetime(raw["checkout"], format="%Y-%m-%d")
        # Extract month, weekday for checkin and checkout, and calculate trip length in Pandas
        raw["mn"] = raw["checkin"].dt.month
        raw["dy1"] = raw["checkin"].dt.weekday
        raw["dy2"] = raw["checkout"].dt.weekday
        raw["length"] = np.log1p((raw["checkout"] - raw["checkin"]).dt.days)
        # Extract first checkin and last checkout for each trip in Pandas
        tmpD = raw[raw["dcount"] == 0][["utrip_id", "checkin"]]
        tmpD.columns = ["utrip_id", "first_checkin"]
        raw = raw.merge(tmpD, on="utrip_id", how="left")
        tmpD = raw[raw["icount"] == 0][["utrip_id", "checkout"]]
        tmpD.columns = ["utrip_id", "last_checkout"]
        raw = raw.merge(tmpD, on="utrip_id", how="left")
        # Calculate trip length and derive last checkin and first checkout in Pandas
        raw["trip_length"] = (raw["last_checkout"] - raw["first_checkin"]).dt.days
        raw["trip_length"] = np.log1p(np.abs(raw["trip_length"])) * np.sign(raw["trip_length"])
        tmpD = raw[raw["icount"] == 0][["utrip_id", "checkin"]]
        tmpD.columns = ["utrip_id", "last_checkin"]
        raw = raw.merge(tmpD, on="utrip_id", how="left")
        tmpD = raw[raw["dcount"] == 0][["utrip_id", "checkout"]]
        tmpD.columns = ["utrip_id", "first_checkout"]
        raw = raw.merge(tmpD, on="utrip_id", how="left")
        raw["trip_length"] = raw["trip_length"] - raw["trip_length"].mean()
        # Engineer checkout lag and calculate lapse in Pandas
        raw["checkout_lag1"] = raw.groupby("utrip_id_")["checkout"].shift(1, fill_value=None)
        raw["lapse"] = (raw["checkin"] - raw["checkout_lag1"]).dt.days.fillna(-1)
        # Engineer weekend and season features in Pandas
        raw["day_name"] = raw["checkin"].dt.weekday
        raw["weekend"] = raw["day_name"].isin([5, 6]).astype("int8")
        df_season = pd.DataFrame({"mn": range(1, 13), "season": ([0] * 3) + ([1] * 3) + ([2] * 3) + ([3] * 3)})
        raw = raw.merge(df_season, how="left", on="mn")
        raw["N"] = raw["N"] - raw["N"].mean()
        raw["N"] /= 3
        raw["log_icount"] = np.log1p(raw["icount"])
        raw["log_dcount"] = np.log1p(raw["dcount"])

        TRAIN_BATCH_SIZE = 1024
        WORKERS = 8
        LR = 1e-3
        EPOCHS = 12
        GRADIENT_ACCUMULATION = 1
        EMBEDDING_DIM = 64
        HIDDEN_DIM = 1024
        DROPOUT_RATE = 0.2
        device = torch.device("mps")

        TRAIN_WITH_TEST = True

        seed = 0
        seed_torch(seed)

        preds_all = []
        best_scores = []
        best_epochs = []
        for fold in range(5):
            preds_fold = []
            print("#" * 25)
            print(f"### FOLD {fold}")
            if TRAIN_WITH_TEST:
                train = raw.loc[
                    (raw.fold != fold) & (raw.dcount > 0) & (raw.istest == 0) | ((raw.istest == 1) & (raw.icount > 0))
                ].copy()
            else:
                train = raw.loc[(raw.fold != fold) & (raw.dcount > 0) & (raw.istest == 0)].copy()
            valid = raw.loc[(raw.fold == fold) & (raw.istest == 0) & (raw.icount == 0) & (raw.reverse == 0)].copy()
            print(train.shape, valid.shape)

            train_dataset = BookingDataset(
                train,
                lag_cities=lag_cities,
                lag_countries=lag_countries,
                target="city_id_",
            )

            train_data_loader = DataLoader(
                train_dataset,
                batch_size=TRAIN_BATCH_SIZE,
                num_workers=WORKERS,
                shuffle=True,
                pin_memory=True,
            )

            valid_dataset = BookingDataset(
                valid,
                lag_cities=lag_cities,
                lag_countries=lag_countries,
                target="city_id_",
            )

            valid_data_loader = DataLoader(
                valid_dataset,
                batch_size=TRAIN_BATCH_SIZE,
                num_workers=WORKERS,
                shuffle=False,
                pin_memory=True,
            )

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
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                pct_start=0.1,
                div_factor=1e3,
                max_lr=3e-3,
                epochs=EPOCHS,
                steps_per_epoch=int(np.ceil(len(train_data_loader) / GRADIENT_ACCUMULATION)),
            )
            best_score = 0
            best_epoch = 0

            for epoch in range(EPOCHS):
                print(time.ctime(), "Epoch:", epoch)
                train_loss = train_epoch(train_data_loader, model, optimizer, scheduler, device)
                val_loss, PREDS, TARGETS = val_epoch(valid_data_loader, model, device)
                PREDS[:, LOW_CITY] = -1e10  # remove low frequency cities
                score = top4(PREDS, TARGETS)

                print(
                    f'Fold {fold} Seed {seed} Ep {epoch} lr {optimizer.param_groups[0]["lr"]:.7f} train loss {np.mean(train_loss):4f} val loss {np.mean(val_loss):4f} score {score:4f}',
                    flush=True,
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    preds_fold = PREDS
                    save_checkpoint(model, optimizer, scheduler, epoch, best_score, fold, seed, fname)
            del model, scheduler, optimizer, valid_data_loader, valid_dataset, train_data_loader, train_dataset
            gc.collect()

            preds_all.append(preds_fold)
            print(f"fold {fold}, best score: {best_score:.6f} best epoch: {best_epoch:3d}")
            best_scores.append(best_score)
            best_epochs.append(best_epoch)
            # with open('../checkpoints/%s/%s_%d_preds.pkl' % (fname, fname, seed), 'wb') as file:
            #    pkl.dump(preds_all, file)

            print()
            for fold, (best_score, best_epoch) in enumerate(zip(best_scores, best_epochs)):
                print(f"fold {fold}, best score: {best_score:.6f} best epoch: {best_epoch:3d}")
            print(f"seed {seed} best score: {best_score:.6f} best epoch: {np.mean(best_epochs):.1f}")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MLPTrain()
