from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


def get_order_in_group(utrip_id_, order):
    for i in range(len(utrip_id_)):
        order[i] = i


def add_cumcount(df, sort_col, outputname):
    df = df.sort_values(sort_col, ascending=True)
    tmp = (
        df[["utrip_id_", "checkin"]]
        .groupby(["utrip_id_"])
        .apply(get_order_in_group, incols=["utrip_id_"], outcols={"order": "int32"})
    )
    tmp.columns = ["utrip_id_", "checkin", outputname]
    df = df.merge(tmp, how="left", on=["utrip_id_", "checkin"])
    df = df.sort_values(sort_col, ascending=True)
    return df


if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "data"
    # Read CSV files using Pandas
    train = pd.read_csv(input_path / "train_set.csv").sort_values(by=["user_id", "checkin"])
    test = pd.read_csv(input_path / "test_set.csv").sort_values(by=["user_id", "checkin"])
    # Add 'istest' column
    train["istest"] = 0
    test["istest"] = 1
    # Concatenate train and test DataFrames
    raw = pd.concat([train, test], sort=False)

    # Sort the DataFrame by 'user_id' and 'checkin'
    raw = raw.sort_values(["user_id", "checkin"], ascending=True, ignore_index=True)

    # Add 'fold' column
    raw["fold"] = 0

    # Use GroupKFold for creating folds
    group_kfold = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(group_kfold.split(X=raw, y=raw, groups=raw["utrip_id"])):
        raw.loc[test_index, "fold"] = fold

    # Display the count of each fold
    raw["fold"].value_counts()
    # Add 'submission' column
    raw["submission"] = 0

    # Set 'submission' to 1 for rows where 'city_id' is 0 and 'istest' is True
    raw.loc[(raw["city_id"] == 0) & (raw["istest"]), "submission"] = 1

    # Number of places visited in each trip
    aggs = raw.groupby("utrip_id", as_index=False)["user_id"].count()
    aggs.columns = ["utrip_id", "N"]
    raw = raw.merge(aggs, on=["utrip_id"], how="inner")

    # Factorize 'utrip_id' and create a mapping
    raw["utrip_id_"], mp = pd.factorize(raw["utrip_id"])

    raw = raw.sort_values(by=["utrip_id_", "checkin"], ascending=True)
    raw = pd.concat([raw, raw.groupby("utrip_id_").cumcount().rename("dcount")], axis=1)
    # Calculate 'icount' column
    raw["icount"] = raw["N"] - raw["dcount"] - 1

    # Save the DataFrame to a CSV file
    raw.to_csv(input_path / "train_and_test_2.csv", index=False)
