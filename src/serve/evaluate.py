"""Score the model on the held-out split.

Metrics go to a metrics.json file as well as the console, because
later an automated workflow will read that file to decide whether
the model is good enough to deploy.
"""

import argparse
import glob
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

TARGET = "Churn"


def main(args):
    model = load_model(args.model_input)
    df = read_data(args.test_data)

    X_test = df.drop(columns=[TARGET])
    y_test = df[TARGET]

    metrics = evaluate(model, X_test, y_test, args.output_dir)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote {metrics_path}")


def load_model(path):
    if os.path.isdir(path):
        return joblib.load(os.path.join(path, "model.pkl"))
    return joblib.load(path)


def read_data(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise RuntimeError(f"No CSV files found in: {path}")
        return pd.concat(
            (pd.read_csv(f) for f in files), ignore_index=True
        )
    return pd.read_csv(path)


def evaluate(model, X_test, y_test, output_dir):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc": float(roc_auc_score(y_test, y_scores)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    os.makedirs(output_dir, exist_ok=True)
    plot_roc_curve(y_test, y_scores, metrics["auc"], output_dir)
    show_confusion_matrix(y_test, y_pred)
    return metrics


def plot_roc_curve(y_test, y_scores, auc, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_scores)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "roc-curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")


def show_confusion_matrix(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True negatives : {tn}")
    print(f"False positives: {fp}   (discount wasted)")
    print(f"False negatives: {fn}   (customer left, no action)")
    print(f"True positives : {tp}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "*" * 60)
    main(parse_args())
    print("*" * 60 + "\n")
    