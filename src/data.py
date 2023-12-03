import torch
from torch.utils.data import Dataset


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
