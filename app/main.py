from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.cache import MemoryCache

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fraud_api")

app = FastAPI(title="Credit Card Fraud Prediction API")

models: dict[str, Any] = {}
thresholds: dict[str, float] = {}
config_lock = Lock()

admin_token = os.getenv("ADMIN_TOKEN", "")
initial_ttl = int(os.getenv("TTL_SECONDS", "600"))
model_n_jobs = int(os.getenv("MODEL_N_JOBS", "1"))

config = {
    "model_version": os.getenv("MODEL_VERSION", "v1"),
    "ttl": initial_ttl,
    "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
    "version_in_key": os.getenv("VERSION_IN_KEY", "true").lower() == "true",
}

cache = MemoryCache(ttl=initial_ttl, maxsize=10_000)
stats = {"hits": 0, "misses": 0, "errors": 0}


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


class ConfigUpdate(BaseModel):
    model_version: str | None = None
    ttl: int | None = Field(default=None, ge=1, le=86_400)
    cache_enabled: bool | None = None
    version_in_key: bool | None = None


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if not admin_token or x_admin_token != admin_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def load_models() -> None:
    for version in ("v1", "v2"):
        model_path = MODEL_DIR / f"fraud_{version}.joblib"
        metadata_path = MODEL_DIR / f"fraud_{version}_meta.json"

        if model_path.exists() and metadata_path.exists():
            model = joblib.load(model_path)

            # Use the same worker setting as the single-worker inference benchmark.
            model.n_jobs = model_n_jobs
            models[version] = model

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            thresholds[version] = float(metadata["metrics"]["threshold"])

    if not models:
        raise RuntimeError("No trained model files found in the models directory.")

    if config["model_version"] not in models:
        raise RuntimeError(
            f"Configured model {config['model_version']} is not available."
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def build_cache_key(payload: dict[str, float], version: str) -> str:
    rounded = {
        name: round(float(value), 6)
        for name, value in payload.items()
    }
    canonical = json.dumps(rounded, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if config["version_in_key"]:
        return f"pred:{version}:{digest}"

    return f"pred:{digest}"


@app.post("/predict")
def predict(transaction: Transaction) -> dict[str, Any]:
    start = time.perf_counter()
    payload = transaction.model_dump()
    active_version = config["model_version"]
    key = build_cache_key(payload, active_version)

    cache_status = "off"
    result = None

    if config["cache_enabled"]:
        try:
            result = cache.get(key)
            if result is None:
                cache_status = "miss"
                stats["misses"] += 1
            else:
                cache_status = "hit"
                stats["hits"] += 1
        except Exception:
            stats["errors"] += 1
            logger.exception("cache_get_failed")
            cache_status = "miss"

    if result is None:
        # Preserve the named columns used when the model was trained.
        row = pd.DataFrame(
            [[payload[name] for name in FEATURES]],
            columns=FEATURES,
        )
        probability = float(models[active_version].predict_proba(row)[0, 1])

        result = {
            "fraud_probability": probability,
            "is_fraud": probability >= thresholds[active_version],
            "model_version": active_version,
        }

        if config["cache_enabled"]:
            try:
                cache.set(key, result)
            except Exception:
                stats["errors"] += 1
                logger.exception("cache_set_failed")

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(json.dumps({
        "event": "prediction",
        "cache": cache_status,
        "server_latency_ms": latency_ms,
        "model_version": result["model_version"],
        "active_model_version": active_version,
        "fraud_probability": result["fraud_probability"],
        "version_in_key": config["version_in_key"],
        "ttl": config["ttl"],
    }))

    return {**result, "cache": cache_status}


@app.get("/admin/stats", dependencies=[Depends(require_admin)])
def get_stats() -> dict[str, Any]:
    return {
        **stats,
        "cache_size": cache.size(),
        "config": dict(config),
    }


@app.post("/admin/flush", dependencies=[Depends(require_admin)])
def flush_cache() -> dict[str, str]:
    cache.flush()
    return {"status": "flushed"}


@app.post("/admin/config", dependencies=[Depends(require_admin)])
def update_config(update: ConfigUpdate) -> dict[str, Any]:
    changes = update.model_dump(exclude_none=True)

    with config_lock:
        requested_version = changes.get("model_version")
        if requested_version is not None and requested_version not in models:
            raise HTTPException(status_code=400, detail="That model is not loaded.")

        requested_ttl = changes.get("ttl")
        if requested_ttl is not None:
            cache.change_ttl(requested_ttl)

        config.update(changes)

    return {"config": dict(config)}