from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "data"
    # Read CSV files using Pandas
    train = pd.read_csv(input_path / "train_set.csv").sort_values(by=["user_id", "checkin"])
    test = pd.read_csv(input_path / "test_set.csv").sort_values(by=["user_id", "checkin"])
    # Add 'istest' column
    train["istest"] = 0
    test["istest"] = 1
    # Concatenate train and test DataFrames
    data = pd.concat([train, test], sort=False)

    # Sort the DataFrame by 'user_id' and 'checkin'
    data = data.sort_values(["user_id", "checkin"], ascending=True, ignore_index=True)

    # Add 'fold' column
    data["fold"] = 0

    # Use GroupKFold for creating folds
    group_kfold = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(group_kfold.split(X=data, y=data, groups=data["utrip_id"])):
        data.loc[test_index, "fold"] = fold

    # Display the count of each fold
    data["fold"].value_counts()
    # Add 'submission' column
    data["submission"] = 0

    # Set 'submission' to 1 for rows where 'city_id' is 0 and 'istest' is True
    data.loc[(data["city_id"] == 0) & (data["istest"]), "submission"] = 1

    # Number of places visited in each trip
    aggs = data.groupby("utrip_id", as_index=False)["user_id"].count()
    aggs.columns = ["utrip_id", "num_visited"]
    data = data.merge(aggs, on=["utrip_id"], how="inner")

    data = data.sort_values(by=["utrip_id", "checkin"], ascending=True)
    data = pd.concat([data, data.groupby("utrip_id").cumcount().rename("order")], axis=1)
    data["inverse_order"] = data["num_visited"] - data["order"] - 1

    # Save the DataFrame to a CSV file
    data.to_csv(input_path / "train_and_test_2.csv", index=False)
