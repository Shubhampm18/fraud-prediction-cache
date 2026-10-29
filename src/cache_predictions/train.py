from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

# train.py is at src/cache_predictions/train.py; parents[2] is the project root.
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
TEST_FRACTION = 0.20


def choose_threshold(
    model: RandomForestClassifier,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
) -> float:
    """Choose an F1-maximizing threshold on a chronological validation slice."""
    probabilities = model.predict_proba(validation_x)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = probabilities >= threshold
        precision = precision_score(validation_y, predictions, zero_division=0)
        recall = recall_score(validation_y, predictions, zero_division=0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def train_one(
    version: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_params: dict,
) -> dict:
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        **model_params,
    )
    model.fit(train_df[FEATURES], train_df[TARGET].astype(int))

    threshold = choose_threshold(
        model,
        validation_df[FEATURES],
        validation_df[TARGET].astype(int),
    )

    test_y = test_df[TARGET].astype(int)
    test_probabilities = model.predict_proba(test_df[FEATURES])[:, 1]
    test_predictions = test_probabilities >= threshold

    metrics = {
        "precision": float(
            precision_score(test_y, test_predictions, zero_division=0)
        ),
        "recall": float(recall_score(test_y, test_predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(test_y, test_probabilities)),
        "threshold": threshold,
        "test_rows": int(len(test_df)),
        "test_fraud_count": int(test_y.sum()),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"fraud_{version}.joblib")

    metadata = {
        "model_version": version,
        "features": FEATURES,
        "metrics": metrics,
        "train_rows": int(len(train_df)),
        "train_fraud_count": int(train_df[TARGET].sum()),
        "validation_rows": int(len(validation_df)),
        "validation_fraud_count": int(validation_df[TARGET].sum()),
        "train_time_min": float(train_df["Time"].min()),
        "train_time_max": float(train_df["Time"].max()),
        "test_time_min": float(test_df["Time"].min()),
        "test_time_max": float(test_df["Time"].max()),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "model_params": model_params,
    }

    (MODEL_DIR / f"fraud_{version}_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {"model": version, **metrics}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"Time", TARGET, *FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    # Chronological split. Time is used for ordering but excluded from FEATURES.
    df = df.sort_values("Time").reset_index(drop=True)

    n_rows = len(df)
    test_start = int(n_rows * (1 - TEST_FRACTION))
    test_df = df.iloc[test_start:].copy()

    v1_end = int(n_rows * 0.50)
    v2_end = test_start

    # The final 10% of each training window is reserved for threshold selection.
    v1_validation_start = int(v1_end * 0.90)
    v2_validation_start = int(v2_end * 0.90)

    v1_train = df.iloc[:v1_validation_start].copy()
    v1_validation = df.iloc[v1_validation_start:v1_end].copy()
    v2_train = df.iloc[:v2_validation_start].copy()
    v2_validation = df.iloc[v2_validation_start:v2_end].copy()

    print(f"Total rows: {n_rows:,}")
    print(f"v1 window: {v1_end:,}; fit rows: {len(v1_train):,}")
    print(f"v2 window: {v2_end:,}; fit rows: {len(v2_train):,}")
    print(f"Shared test rows: {len(test_df):,}")
    print(f"Shared test fraud cases: {int(test_df[TARGET].sum())}")

    if int(test_df[TARGET].sum()) < 50:
        raise ValueError(
            "The shared test period has fewer than 50 fraud cases. "
            "Inspect the time-sorted label distribution before changing the split."
        )

    results = [
        train_one(
            "v1",
            v1_train,
            v1_validation,
            test_df,
            {"n_estimators": 100, "max_depth": 8, "random_state": 1},
        ),
        train_one(
            "v2",
            v2_train,
            v2_validation,
            test_df,
            {"n_estimators": 300, "max_depth": 12, "random_state": 7},
        ),
    ]

    print("\nTest metrics (both models use the same test period):")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()