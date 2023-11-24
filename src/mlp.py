import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import train_epoch, topk, val_epoch, save_checkpoint, BookingDataset, Net, seed_torch, get_device

input_path = Path("../data/")


def _city_count_per_trip(data, low_frequency_city_threshold=9):
    city_count = (
        data.query("istest == 0 and icount > 0")  # train only and exclude last city
        .groupby("city_id")["utrip_id"]
        .count()
        .rename("city_count")
        .reset_index()
    )
    data = data.merge(city_count, how="left", on="city_id")

    # Replace rare cities with -1
    data.loc[data["city_count"] <= low_frequency_city_threshold, "city_id"] = -1
    data = data.sort_values(["utrip_id", "checkin"])

    return data


def _encode_categorical(data, categorical_columns):
    categorical_values = data[categorical_columns].apply(lambda x: x.factorize()[0])
    low_frequency_city_index = categorical_values.loc[data["city_id"] == -1, "city_id"].unique()

    assert len(low_frequency_city_index) == 1

    data[categorical_columns] = categorical_values

    num_cities = data["city_id"].max() + 1
    num_hotels = data["hotel_country"].max() + 1
    num_devices = data["device_class"].max() + 1

    return data, low_frequency_city_index[0], num_cities, num_hotels, num_devices


def _add_reverse_training_data(data):
    data = data.assign(reverse=0)
    reverse_training_data = data.query("istest == 0").copy()
    reverse_training_data = reverse_training_data.assign(
        reverse=1,
        utrip_id=reverse_training_data["utrip_id"] + "_r",
    )
    reverse_training_data[["icount", "dcount"]] = reverse_training_data[["dcount", "icount"]]
    reverse_training_data = reverse_training_data.sort_values(["utrip_id", "dcount"], ignore_index=True)
    data = pd.concat([data, reverse_training_data], ignore_index=True)

    data["sorting"] = np.arange(data.shape[0])

    return data


def _lag_cities_countries(data, NUM_CITIES, NUM_HOTELS, n_lags=5):
    lag_cities = data.groupby("utrip_id")["city_id"].shift(range(1, n_lags + 1), fill_value=NUM_CITIES)
    lag_cities_columns = [f"city_id_lag{i}" for i in range(1, n_lags + 1)]
    lag_cities.columns = lag_cities_columns

    lag_countries = data.groupby("utrip_id")["hotel_country"].shift(range(1, n_lags + 1), fill_value=NUM_HOTELS)
    lag_countries_columns = [f"hotel_id_lag{i}" for i in range(1, n_lags + 1)]
    lag_countries.columns = lag_countries_columns

    data = pd.concat([data, lag_cities, lag_countries], axis=1)

    return data, lag_cities_columns, lag_countries_columns


def _find_first_last(data, first_columns=None, last_columns=None, first_names=None, last_names=None):
    if first_columns is not None:
        first = data.query("dcount == 0")[["utrip_id"] + first_columns]
        first.columns = ["utrip_id"] + first_names
        data = data.merge(first, on="utrip_id", how="left")

    if last_columns is not None:
        last = data.query("icount == 0")[["utrip_id"] + last_columns]
        last.columns = ["utrip_id"] + last_names
        data = data.merge(last, on="utrip_id", how="left")

    return data


def load_data(n_trips=None):
    LAGS = 5

    data = pd.read_csv(input_path / "train_and_test_2.csv")
    if n_trips:
        data = data.query("utrip_id_ <= @n_trips").reset_index(drop=True)

    # Replace 0s in 'city_id' with NaN
    data.loc[data["city_id"] == 0, "city_id"] = np.NaN
    data = _city_count_per_trip(data, low_frequency_city_threshold=9)

    # Reverse the data for training set
    data = _add_reverse_training_data(data)

    # Factorize categorical columns
    data, LOW_CITY, NUM_CITIES, NUM_HOTELS, NUM_DEVICE = _encode_categorical(
        data, ["utrip_id", "city_id", "hotel_country", "booker_country", "device_class"]
    )

    data, lag_cities, lag_countries = _lag_cities_countries(data, NUM_CITIES, NUM_HOTELS, n_lags=LAGS)

    # Extract the first city and country for each trip
    data = _find_first_last(
        data, first_columns=["city_id", "hotel_country"], first_names=["first_city", "first_country"]
    )

    # Convert 'checkin' and 'checkout' columns to datetime in Pandas
    data["checkin"] = pd.to_datetime(data["checkin"], format="%Y-%m-%d")
    data["checkout"] = pd.to_datetime(data["checkout"], format="%Y-%m-%d")

    # Extract first checkin and last checkout for each trip in Pandas
    data = _find_first_last(
        data,
        first_columns=["checkin", "checkout"],
        first_names=["first_checkin", "first_checkout"],
        last_columns=["checkin", "checkout"],
        last_names=["last_checkin", "last_checkout"],
    )

    # Extract month, weekday for checkin and checkout, and calculate trip length in Pandas
    data["mn"] = data["checkin"].dt.month
    data["dy1"] = data["checkin"].dt.weekday
    data["dy2"] = data["checkout"].dt.weekday
    data["day_name"] = data["checkin"].dt.weekday
    data["weekend"] = data["day_name"].isin([5, 6]).astype("int8")
    df_season = pd.DataFrame({"mn": range(1, 13), "season": ([0] * 3) + ([1] * 3) + ([2] * 3) + ([3] * 3)})
    data = data.merge(df_season, how="left", on="mn")
    data["length"] = np.log1p((data["checkout"] - data["checkin"]).dt.days)
    data["trip_length"] = (data["last_checkout"] - data["first_checkin"]).dt.days
    data["trip_length"] = np.log1p(np.abs(data["trip_length"])) * np.sign(data["trip_length"])
    data["trip_length"] = data["trip_length"] - data["trip_length"].mean()

    # Engineer checkout lag and calculate lapse in Pandas
    data["checkout_lag1"] = data.groupby("utrip_id")["checkout"].shift(1, fill_value=None)
    data["lapse"] = (data["checkin"] - data["checkout_lag1"]).dt.days.fillna(-1)
    # Engineer weekend and season features in Pandas
    data["N"] = data["N"] - data["N"].mean()
    data["N"] /= 3
    data["log_icount"] = np.log1p(data["icount"])
    data["log_dcount"] = np.log1p(data["dcount"])

    return data, lag_cities, lag_countries, NUM_CITIES, NUM_HOTELS, NUM_DEVICE, LOW_CITY


def train(raw, lag_cities, lag_countries, NUM_CITIES, NUM_HOTELS, NUM_DEVICE, LOW_CITY):
    fname = "mlp"

    TRAIN_BATCH_SIZE = 1024
    LR = 1e-3
    EPOCHS = 12
    GRADIENT_ACCUMULATION = 1
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 1024
    DROPOUT_RATE = 0.2
    TRAIN_WITH_TEST = True

    device = get_device()
    print("device:", device)

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
            target="city_id",
        )

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        valid_dataset = BookingDataset(
            valid,
            lag_cities=lag_cities,
            lag_countries=lag_countries,
            target="city_id",
        )

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=TRAIN_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=8,
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
            score = topk(PREDS, TARGETS)

            print(
                f"""
                Fold {fold} 
                Seed {seed} 
                Ep {epoch} 
                lr {optimizer.param_groups[0]["lr"]:.7f} 
                train loss {np.mean(train_loss):4f} 
                val loss {np.mean(val_loss):4f} 
                score {score:4f}""",
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

        print()
        for fold, (best_score, best_epoch) in enumerate(zip(best_scores, best_epochs)):
            print(f"fold {fold}, best score: {best_score:.6f} best epoch: {best_epoch:3d}")
        print(f"seed {seed} best score: {best_score:.6f} best epoch: {np.mean(best_epochs):.1f}")


if __name__ == "__main__":
    raw, lag_cities, lag_countries, NUM_CITIES, NUM_HOTELS, NUM_DEVICE, LOW_CITY = load_data()
    train(raw, lag_cities, lag_countries, NUM_CITIES, NUM_HOTELS, NUM_DEVICE, LOW_CITY)
