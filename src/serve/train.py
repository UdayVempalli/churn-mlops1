"""Train the churn model.

Encoding and scaling live inside a scikit-learn Pipeline, not in the
prep step, so the transformers are fitted on the training split only
and travel with the model at serving time.
"""

import argparse
import glob
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main(args):
    df = read_data(args.training_data)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = build_pipeline(X_train, reg_rate=args.reg_rate)

    print("Training model...")
    model.fit(X_train, y_train)
    print(f"Training accuracy: {model.score(X_train, y_train):.4f}")

    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Wrote {model_path}")

    os.makedirs(args.test_data_output, exist_ok=True)
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_path = os.path.join(args.test_data_output, "test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Wrote {test_path}")


def read_data(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise RuntimeError(f"No CSV files found in: {path}")
        return pd.concat(
            (pd.read_csv(f) for f in files), ignore_index=True
        )
    return pd.read_csv(path)


def split_data(df):
    """Stratified split: same churn rate in both halves."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_pipeline(X, reg_rate=0.1):
    """Preprocessing plus estimator, as one fitted object."""
    numeric_features = (
        X.select_dtypes(include="number").columns.tolist()
    )
    categorical_features = (
        X.select_dtypes(exclude="number").columns.tolist()
    )

    numeric_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_steps, numeric_features),
            ("categorical", categorical_steps, categorical_features),
        ]
    )

    estimator = LogisticRegression(
        C=1 / reg_rate,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[("preprocess", preprocessor), ("model", estimator)]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--reg_rate", type=float, default=0.1)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument(
        "--test_data_output", type=str, required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "*" * 60)
    main(parse_args())
    print("*" * 60 + "\n")