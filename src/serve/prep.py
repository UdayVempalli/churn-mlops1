"""Clean the raw churn data.

Cleaning only. Encoding and scaling live inside the model pipeline
in train.py, so they are fitted on training data alone.
"""

import argparse
import glob
import os

import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def main(args):
    df = read_data(args.input_data)
    print(f"Read {len(df)} rows, {len(df.columns)} columns")

    df = clean_data(df)
    print(f"After cleaning: {len(df)} rows")
    print(f"Churn rate: {df[TARGET].mean():.1%}")

    os.makedirs(args.output_data, exist_ok=True)
    out_path = os.path.join(args.output_data, "churn-clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def read_data(path):
    """Read a single CSV, or every CSV in a folder."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise RuntimeError(f"No CSV files found in: {path}")
        return pd.concat(
            (pd.read_csv(f) for f in files), ignore_index=True
        )
    return pd.read_csv(path)


def clean_data(df):
    """Apply the four fixes found during exploration."""
    df = df.copy()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0.0)

    return df.drop_duplicates()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "*" * 60)
    main(parse_args())
    print("*" * 60 + "\n")