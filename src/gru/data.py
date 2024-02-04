from pathlib import Path

import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset

input_path = Path("../../data/")


class BookingDataset(Dataset):
    def __init__(
        self,
        data,
        lag_cities,
        target=None,
    ):
        super(BookingDataset, self).__init__()
        self.lag_cities = data[lag_cities].to_numpy()
        self.month = data["month"].to_numpy()
        self.checkin_day = data["checkin_day"].to_numpy()
        self.checkout_day = data["checkout_day"].to_numpy()
        self.stay_length = data["stay_length"].to_numpy()
        self.trip_length = data["trip_length"].to_numpy()
        self.lapse = data["lapse"].to_numpy()
        self.season = data["season"].to_numpy()
        self.weekend = data["weekend"].to_numpy()
        self.country_lag_1 = data["country_lag_1"].to_numpy()
        self.first_city = data["first_city"].to_numpy()
        self.booker_country = data["booker_country"].to_numpy()
        self.device_class = data["device_class"].to_numpy()
        self.affiliate_id = data["affiliate_id"].to_numpy()
        self.order = data["order"].to_numpy()
        self.inverse_order = data["inverse_order"].to_numpy()

        if target is None:
            self.target = None
        else:
            self.target = data[target].to_numpy()

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
            "lapse": torch.tensor([self.lapse[idx]], dtype=torch.float),
            "season": torch.tensor([self.season[idx]], dtype=torch.long),
            "weekend": torch.tensor([self.weekend[idx]], dtype=torch.long),
            "country_lag_1": torch.tensor([self.country_lag_1[idx]], dtype=torch.long),
            "first_city": torch.tensor([self.first_city[idx]], dtype=torch.long),
            "booker_country": torch.tensor([self.booker_country[idx]], dtype=torch.long),
            "device_class": torch.tensor([self.device_class[idx]], dtype=torch.long),
            "affiliate_id": torch.tensor([self.affiliate_id[idx]], dtype=torch.long),
            "order": torch.tensor([self.order[idx]], dtype=torch.float),
            "inverse_order": torch.tensor([self.inverse_order[idx]], dtype=torch.float),
        }
        if self.target is not None:
            input_dict["target"] = torch.tensor([self.target[idx]], dtype=torch.long)
        return input_dict


class BookingData:
    NUM_LAGS = 5
    LOW_FREQUENCY_CITY_THRESHOLD = 9
    INPUT_PATH = input_path / "train_and_test_2.csv"

    def __init__(self):
        self.num_cities = None
        self.num_countries = None
        self.num_devices = None
        self.data = self.process()

    def process(self):
        data = pl.read_csv(self.INPUT_PATH)
        data = data.with_columns(pl.col("city_id").replace(0, None))

        # add city count
        city_count = (
            pl.when(pl.col("istest").eq(0).or_(pl.col("inverse_order").gt(0)))
            .then(pl.count("utrip_id").over("city_id"))
            .otherwise(None)
            .alias("city_count")
        )
        city_id = (
            pl.when(city_count.le(self.LOW_FREQUENCY_CITY_THRESHOLD))
            .then(-1)
            .otherwise(pl.col("city_id"))
            .alias("city_id")
        )
        data = data.with_columns(city_count, city_id)
        reverse_data = data.filter(pl.col("istest").eq(0))
        reverse_data = reverse_data.with_columns(
            pl.lit(1).alias("reverse"),
            pl.col("utrip_id").add("_r"),
            pl.col("order").alias("inverse_order"),
            pl.col("inverse_order").alias("order"),
        )

        data = pl.concat([data.with_columns(pl.lit(0).alias("reverse")), reverse_data], how="vertical")
        data = data.with_row_index(name="sorting")
        data = data.with_columns(
            pl.col(
                [
                    "utrip_id",
                    "city_id",
                    "hotel_country",
                    "booker_country",
                    "device_class",
                    "affiliate_id",
                ]
            ).map_batches(lambda x: pd.factorize(x.to_numpy())[0])
        )

        self.num_cities = data["city_id"].max() + 1
        self.num_countries = data["hotel_country"].max() + 1
        self.num_devices = data["device_class"].max() + 1

        data = data.sort(by=["utrip_id", "order"], descending=False)
        city_lags = [
            pl.col("city_id").shift(n, fill_value=-1).over("utrip_id").alias(f"city_id_lag_{n}")
            for n in range(1, self.NUM_LAGS + 1)
        ]
        country_lag_1 = pl.col("hotel_country").shift(1, fill_value=-1).over("utrip_id").alias(f"country_lag_1")

        first_city = pl.first("city_id").over("utrip_id").alias("first_city")

        checkin_col = pl.col("checkin").str.to_date()
        checkout_col = pl.col("checkout").str.to_date()
        month = checkin_col.dt.month().sub(1).alias("month")
        checkin_day = checkin_col.dt.weekday().sub(1).alias("checkin_day")
        checkout_day = checkout_col.dt.weekday().sub(1).alias("checkout_day")
        weekend = checkin_day.is_in([5, 6]).alias("weekend")
        season = (month // 3).alias("season")
        stay_length = (checkout_col - checkin_col).dt.total_days().alias("stay_length")

        first_checkin = checkin_col.first().over("utrip_id").alias("first_checkin")
        last_checkout = checkout_col.last().over("utrip_id").alias("last_checkout")
        trip_length = (last_checkout - first_checkin).dt.total_days().alias("trip_length")

        checkout_lag1 = checkout_col.shift(1, fill_value=None).over("utrip_id").alias("checkout_lag1")
        lapse = (checkin_col - checkout_lag1).dt.total_days().fill_null(-1).alias("lapse")

        data = data.with_columns(
            city_lags
            + [
                country_lag_1,
                first_city,
                checkin_day,
                checkout_day,
                weekend,
                month,
                season,
                stay_length,
                trip_length,
                lapse,
            ]
        )

        return data

    def get_train_dataset(self, fold, train_with_test):
        filter = pl.col("fold").ne(fold).and_(pl.col("order").gt(0)).and_(pl.col("istest").eq(0))
        if train_with_test:
            filter = filter.or_(pl.col("istest").eq(1).and_(pl.col("inverse_order").gt(0)))

        train = self.data.filter(filter)

        return BookingDataset(
            train,
            lag_cities=[f"city_id_lag_{n}" for n in range(1, self.NUM_LAGS + 1)],
            target="city_id",
        )

    def get_valid_dataset(self, fold):
        filter = (
            pl.col("fold")
            .eq(fold)
            .and_(pl.col("istest").eq(0))
            .and_(pl.col("inverse_order").eq(0))
            .and_(pl.col("reverse").eq(0))
        )
        valid = self.data.filter(filter)

        return BookingDataset(
            valid,
            lag_cities=[f"city_id_lag_{n}" for n in range(1, self.NUM_LAGS + 1)],
            target="city_id",
        )
