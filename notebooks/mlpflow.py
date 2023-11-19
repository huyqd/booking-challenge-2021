import gc
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from metaflow import FlowSpec, step, batch, IncludeFile
from torch.utils.data import DataLoader

from notebooks.mlp_utils import train_epoch, top4, val_epoch, save_checkpoint, BookingDataset, Net, seed_torch

input_path = Path("../data/")


class MLPFlow(FlowSpec):
    data = IncludeFile("data", help="data file", is_text=False, default=str(input_path / "train_and_test_2.csv"))

    @step
    def start(self):
        self.next(self.train)

    @batch(
        gpu=1,
        cpu=8,
        memory=16 * 1024,
        image="475126315063.dkr.ecr.eu-central-1.amazonaws.com/metaflow-ltr:gpu",
    )
    @step
    def train(self):
        fname = "mlp"

        LOW_CITY_THR = 9
        LAGS = 5
        # Read CSV using Pandas
        raw = pd.read_csv(BytesIO(self.data))
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
            raw[f"country_lag{i}"] = raw.groupby("utrip_id_")["hotel_country_"].shift(i, fill_value=NUM_HOTELS)
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
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

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
    MLPFlow()
