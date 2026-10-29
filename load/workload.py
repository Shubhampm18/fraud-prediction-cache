from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TEST_FRACTION = 0.20
POOL_FRAUDS = 50
POOL_LEGITIMATE = 450
REQUEST_COUNT = 1000
SEED = 42


def create_pool(seed: int = SEED) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).sort_values("Time").reset_index(drop=True)

    test_start = int(len(df) * (1 - TEST_FRACTION))
    test_df = df.iloc[test_start:]

    fraud_rows = test_df[test_df["Class"] == 1]
    legitimate_rows = test_df[test_df["Class"] == 0]

    if len(fraud_rows) < POOL_FRAUDS:
        raise ValueError(
            f"Test period has {len(fraud_rows)} fraud rows; "
            f"{POOL_FRAUDS} are required."
        )
    if len(legitimate_rows) < POOL_LEGITIMATE:
        raise ValueError(
            f"Test period has {len(legitimate_rows)} legitimate rows; "
            f"{POOL_LEGITIMATE} are required."
        )

    fraud = fraud_rows.sample(n=POOL_FRAUDS, random_state=seed)
    legitimate = legitimate_rows.sample(n=POOL_LEGITIMATE, random_state=seed)

    pool = pd.concat([fraud, legitimate])
    return pool.sample(frac=1, random_state=seed).reset_index(drop=True)


def create_sequence(
    pool_size: int,
    request_count: int = REQUEST_COUNT,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Zipf-like repeat pattern: lower pool indices are sampled more often.
    ranks = np.arange(1, pool_size + 1, dtype=float)
    weights = 1 / ranks
    weights /= weights.sum()

    return rng.choice(pool_size, size=request_count, p=weights)


def save_design(pool: pd.DataFrame, sequence: np.ndarray) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(RESULTS_DIR / "workload_pool.csv", index=False)
    pd.DataFrame({"pool_index": sequence}).to_csv(
        RESULTS_DIR / "workload_sequence.csv",
        index=False,
    )


def load_or_create_design() -> tuple[pd.DataFrame, np.ndarray]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pool_path = RESULTS_DIR / "workload_pool.csv"
    sequence_path = RESULTS_DIR / "workload_sequence.csv"

    if pool_path.exists():
        pool = pd.read_csv(pool_path)
    else:
        pool = create_pool()

    if sequence_path.exists():
        sequence = pd.read_csv(sequence_path)["pool_index"].to_numpy(dtype=int)

        # Reuse the sequence only when it has the requested length and valid indices.
        if (
            len(sequence) == REQUEST_COUNT
            and np.all(sequence >= 0)
            and np.all(sequence < len(pool))
        ):
            return pool, sequence

    sequence = create_sequence(
        pool_size=len(pool),
        request_count=REQUEST_COUNT,
        seed=SEED,
    )
    save_design(pool, sequence)
    return pool, sequence


def run(api_url: str, run_name: str, pause_seconds: float) -> None:
    pool, sequence = load_or_create_design()
    records: list[dict] = []

    with requests.Session() as session:
        for request_number, pool_index in enumerate(sequence):
            row = pool.iloc[int(pool_index)]
            payload = {feature: float(row[feature]) for feature in FEATURES}

            started = time.perf_counter()
            response = session.post(
                f"{api_url.rstrip('/')}/predict",
                json=payload,
                timeout=30,
            )
            client_latency_ms = (time.perf_counter() - started) * 1000

            response.raise_for_status()
            body = response.json()

            records.append({
                "run": run_name,
                "request_number": request_number,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pool_index": int(pool_index),
                "actual_class": int(row["Class"]),
                "client_latency_ms": client_latency_ms,
                "cache": body["cache"],
                "model_version": body["model_version"],
                "fraud_probability": body["fraud_probability"],
                "is_fraud": body["is_fraud"],
            })

            if pause_seconds > 0:
                time.sleep(pause_seconds)

    output_path = RESULTS_DIR / f"{run_name}.csv"
    pd.DataFrame(records).to_csv(output_path, index=False)

    print(f"Requests recorded: {len(records)}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base API URL")
    parser.add_argument("--name", required=True, help="Run name, such as run_ttl_30")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to wait between requests",
    )
    args = parser.parse_args()

    run(
        api_url=args.url,
        run_name=args.name,
        pause_seconds=args.pause,
    )