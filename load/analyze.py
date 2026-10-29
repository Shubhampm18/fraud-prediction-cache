from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def summarize_run(path: Path) -> dict:
    df = pd.read_csv(path)

    hits = df.loc[df["cache"] == "hit", "client_latency_ms"]
    misses = df.loc[df["cache"] == "miss", "client_latency_ms"]

    return {
        "run": path.stem,
        "requests": len(df),
        "hit_rate": float((df["cache"] == "hit").mean()),
        "latency_mean_ms": float(df["client_latency_ms"].mean()),
        "latency_p50_ms": float(df["client_latency_ms"].quantile(0.50)),
        "latency_p95_ms": float(df["client_latency_ms"].quantile(0.95)),
        "latency_p99_ms": float(df["client_latency_ms"].quantile(0.99)),
        "hit_p50_ms": float(hits.quantile(0.50)) if len(hits) else float("nan"),
        "hit_p95_ms": float(hits.quantile(0.95)) if len(hits) else float("nan"),
        "miss_p50_ms": float(misses.quantile(0.50)) if len(misses) else float("nan"),
        "miss_p95_ms": float(misses.quantile(0.95)) if len(misses) else float("nan"),
    }


def main() -> None:
    paths = sorted(RESULTS_DIR.glob("run_*.csv"))
    if not paths:
        raise SystemExit(f"No run_*.csv files found in {RESULTS_DIR}")

    summary = pd.DataFrame(summarize_run(path) for path in paths)
    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ax = summary.plot(
        x="run",
        y="hit_rate",
        kind="bar",
        legend=False,
        figsize=(9, 4),
        title="Cache hit rate by run",
    )
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hit_rate.png", dpi=160)
    plt.close()

    ax = summary.plot(
        x="run",
        y=["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"],
        kind="bar",
        figsize=(10, 5),
        title="Client latency by run",
    )
    ax.set_ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "latency_by_run.png", dpi=160)
    plt.close()

    print(f"\nSaved summary and plots under {RESULTS_DIR}")


if __name__ == "__main__":
    main()