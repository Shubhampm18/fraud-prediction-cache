from __future__ import annotations

import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TEST_FRACTION = 0.20
WARMUP_CALLS = 10
MEASURED_CALLS = 200
SEED = 42


def main() -> None:
    df = pd.read_csv(DATA_PATH).sort_values("Time").reset_index(drop=True)

    test_start = int(len(df) * (1 - TEST_FRACTION))
    test_df = df.iloc[test_start:]

    rng = np.random.default_rng(SEED)
    row_indices = rng.integers(0, len(test_df), size=MEASURED_CALLS)

    # Keep feature names, as used when the models were trained.
    sample_rows = [
        test_df.iloc[[i]][FEATURES]
        for i in row_indices
    ]

    results = []

    for version in ("v1", "v2"):
        model_path = MODEL_DIR / f"fraud_{version}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}")

        model = joblib.load(model_path)

        # Single-row prediction on one CPU worker avoids parallel scheduling
        # overhead. This changes only this in-memory benchmark instance.
        model.n_jobs = 1

        for i in range(WARMUP_CALLS):
            model.predict_proba(sample_rows[i % len(sample_rows)])

        durations_ms = []
        for row in sample_rows:
            started = time.perf_counter()
            model.predict_proba(row)
            durations_ms.append((time.perf_counter() - started) * 1000)

        results.append({
            "model": version,
            "calls": len(durations_ms),
            "mean_ms": float(np.mean(durations_ms)),
            "p50_ms": float(np.percentile(durations_ms, 50)),
            "p95_ms": float(np.percentile(durations_ms, 95)),
            "p99_ms": float(np.percentile(durations_ms, 99)),
            "min_ms": float(np.min(durations_ms)),
            "max_ms": float(np.max(durations_ms)),
            "n_jobs": 1,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "baseline_inference.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"Python: {platform.python_version()}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Test-period rows available: {len(test_df):,}")
    print(f"Warm-up calls per model: {WARMUP_CALLS}")
    print(f"Measured calls per model: {MEASURED_CALLS}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()