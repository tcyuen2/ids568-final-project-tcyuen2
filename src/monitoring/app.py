"""
FastAPI prediction service for the Iris classifier from M3.

Endpoints:
  POST /predict   - Accept 4 raw Iris features, return predicted class + probability
  GET  /metrics   - Prometheus scrape endpoint
  GET  /health    - Liveness check
  GET  /          - Service info

This service is the *instrumented production surface* of the M3 model.
It loads pickled artifacts (model.pkl, scaler.pkl) bundled in models/,
mirrors M3's preprocessing (interaction features + scaling), and emits
metrics defined in instrumentation.py to a Prometheus scrape endpoint.

Design choices:
  - Drift PSI is recomputed in a background task every 10 seconds, not on
    every request. Keeps p99 latency clean while ensuring metrics stay fresh
    relative to a typical 15s Prometheus scrape interval.
  - Input integrity uses StandardScaler-aware bounds: post-scaling values
    should be near N(0,1), so |z| > 5 is a strong anomaly signal regardless
    of which underlying feature it is.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .drift import RollingWindow, histogram_pcts, psi
from .instrumentation import (
    FEATURE_DRIFT_PSI,
    FEATURE_VALUE_MEAN,
    FEATURE_VALUE_STD,
    INFERENCE_DURATION_SECONDS,
    INPUT_ANOMALIES_TOTAL,
    MODEL_INFO,
    PREDICTION_CONFIDENCE,
    PREDICTION_DRIFT_PSI,
    PREDICTIONS_TOTAL,
    REQUEST_DURATION_SECONDS,
    REQUESTS_IN_FLIGHT,
    REQUESTS_TOTAL,
)
from .reference import Reference

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# Paths resolve relative to the repository root, regardless of how the
# service is launched (uvicorn from src/, Docker, etc.).
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "model.pkl"
SCALER_PATH = REPO_ROOT / "models" / "scaler.pkl"
REFERENCE_PATH = REPO_ROOT / "models" / "reference_stats.json"

# Lineage metadata. Update these when promoting a new model version.
MLFLOW_RUN_ID = "6e01be84da0e47d68b1fc5caa9749e40"
MODEL_VERSION = "v1.0.0"

# Rolling-window capacity for drift detection. Larger = more stable PSI
# estimates but slower to detect actual drift. 500 = ~3 minutes of history
# at 3 req/s, which is a reasonable balance for this teaching service.
DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "500"))

# How often to recompute drift PSI (seconds). Should be << Prometheus
# scrape interval to ensure scrapes pull recent values.
DRIFT_RECOMPUTE_INTERVAL_S = float(os.getenv("DRIFT_RECOMPUTE_INTERVAL_S", "10"))

# Anomaly threshold on post-scaling values. After StandardScaler, training
# data is ~N(0,1), so |z| > 5 is a 1-in-3.5M event under normality —
# overwhelmingly likely a malformed input.
ANOMALY_Z_THRESHOLD = 5.0


# --------------------------------------------------------------------------- #
# Request / response schemas                                                  #
# --------------------------------------------------------------------------- #


class PredictRequest(BaseModel):
    """Iris features in their natural (raw, unscaled) units (cm)."""

    sepal_length: float = Field(..., description="Sepal length in cm.")
    sepal_width: float = Field(..., description="Sepal width in cm.")
    petal_length: float = Field(..., description="Petal length in cm.")
    petal_width: float = Field(..., description="Petal width in cm.")


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    probabilities: dict[str, float]
    confidence: float
    inference_ms: float


# --------------------------------------------------------------------------- #
# Service state                                                               #
# --------------------------------------------------------------------------- #


class ServiceState:
    """Holds the loaded model, scaler, reference stats, and rolling window.

    Encapsulated so tests can inject mock state without monkey-patching
    module-level globals.
    """

    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        self.reference = Reference(REFERENCE_PATH)
        self.window = RollingWindow(
            feature_names=self.reference.feature_names,
            capacity=DRIFT_WINDOW_SIZE,
        )
        logger.info(
            "Loaded model=%s scaler=%s reference=%s features=%d",
            MODEL_PATH.name,
            SCALER_PATH.name,
            REFERENCE_PATH.name,
            len(self.reference.feature_names),
        )


state: ServiceState | None = None


# --------------------------------------------------------------------------- #
# Background drift recomputation                                              #
# --------------------------------------------------------------------------- #


async def drift_recompute_loop():
    """Periodically recomputes per-feature PSI, mean, std from the rolling
    window and updates the corresponding Prometheus gauges."""
    while True:
        try:
            await asyncio.sleep(DRIFT_RECOMPUTE_INTERVAL_S)
            if state is None or state.window.size() == 0:
                continue
            recompute_drift_metrics()
        except asyncio.CancelledError:
            logger.info("Drift recompute loop cancelled.")
            raise
        except Exception:
            # Never let the loop die — log and keep going.
            logger.exception("Drift recompute failed; continuing.")


def recompute_drift_metrics() -> None:
    """Pull a snapshot from the rolling window, compute drift stats,
    and push them to Prometheus gauges."""
    assert state is not None
    feature_snap = state.window.snapshot_features()
    pred_snap = state.window.snapshot_predictions()

    # Per-feature: mean, std, PSI vs. training reference
    for name, values in feature_snap.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        FEATURE_VALUE_MEAN.labels(feature=name).set(float(arr.mean()))
        FEATURE_VALUE_STD.labels(feature=name).set(float(arr.std()))

        ref = state.reference.feature(name)
        actual_pcts = histogram_pcts(values, ref.bin_edges)
        FEATURE_DRIFT_PSI.labels(feature=name).set(
            psi(np.array(ref.bin_pcts), actual_pcts)
        )

    # Output drift: PSI of class distribution
    if pred_snap:
        n_classes = len(state.reference.class_names)
        class_counts = np.bincount(pred_snap, minlength=n_classes)
        actual_class_pcts = class_counts / class_counts.sum()
        expected_class_pcts = np.array(state.reference.class_pcts_array())
        PREDICTION_DRIFT_PSI.set(psi(expected_class_pcts, actual_class_pcts))


# --------------------------------------------------------------------------- #
# Lifespan: startup / shutdown                                                #
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = ServiceState()
    MODEL_INFO.labels(
        model_version=MODEL_VERSION, mlflow_run_id=MLFLOW_RUN_ID
    ).set(1)
    drift_task = asyncio.create_task(drift_recompute_loop())
    logger.info("Service started.")
    try:
        yield
    finally:
        drift_task.cancel()
        try:
            await drift_task
        except asyncio.CancelledError:
            pass
        logger.info("Service stopped.")


app = FastAPI(
    title="Iris Classifier - Production Monitoring Demo",
    description="Instrumented serving layer for the M3 Random Forest model.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Middleware                                                                  #
# --------------------------------------------------------------------------- #


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Tracks request count, in-flight gauge, and request duration histogram."""
    # /metrics itself shouldn't appear in latency stats — it's pulled by
    # Prometheus on a tight loop and would dominate the histogram.
    is_scrape = request.url.path == "/metrics"

    if not is_scrape:
        REQUESTS_IN_FLIGHT.inc()
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        if not is_scrape:
            REQUESTS_TOTAL.labels(
                endpoint=endpoint, status_code=str(status_code)
            ).inc()
            REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(elapsed)
            REQUESTS_IN_FLIGHT.dec()


# --------------------------------------------------------------------------- #
# Endpoints                                                                   #
# --------------------------------------------------------------------------- #


@app.get("/")
def root():
    return {
        "service": "iris-classifier",
        "model_version": MODEL_VERSION,
        "mlflow_run_id": MLFLOW_RUN_ID,
        "endpoints": ["/predict", "/metrics", "/health"],
    }


@app.get("/health")
def health():
    if state is None:
        raise HTTPException(503, "Service not initialized")
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run inference on a single Iris example.

    Pipeline:
      1. Validate inputs for NaN / inf (Pydantic accepts them by default).
      2. Compute interaction features (matches M3 preprocessing exactly).
      3. Apply StandardScaler (loaded from M3 training).
      4. Run model.predict + predict_proba.
      5. Update integrity, prediction, and rolling-window metrics.
    """
    assert state is not None

    raw = [req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]

    # ----- Integrity check 1: NaN / inf in raw inputs ----- #
    raw_feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for name, val in zip(raw_feature_names, raw):
        if math.isnan(val):
            INPUT_ANOMALIES_TOTAL.labels(feature=name, anomaly_type="nan").inc()
            raise HTTPException(400, f"Feature '{name}' is NaN")
        if math.isinf(val):
            INPUT_ANOMALIES_TOTAL.labels(feature=name, anomaly_type="inf").inc()
            raise HTTPException(400, f"Feature '{name}' is infinite")

    # ----- Compute interactions (must match M3 preprocessing) ----- #
    sepal_l_x_sepal_w = raw[0] * raw[1]
    petal_l_x_petal_w = raw[2] * raw[3]
    feature_vec = np.array([raw + [sepal_l_x_sepal_w, petal_l_x_petal_w]])

    # ----- Scale ----- #
    scaled = state.scaler.transform(feature_vec)[0]

    # ----- Integrity check 2: extreme z-scores after scaling ----- #
    # Naming order must match Reference.feature_names.
    feature_names = state.reference.feature_names
    scaled_dict = dict(zip(feature_names, scaled.tolist()))
    for name, val in scaled_dict.items():
        if abs(val) > ANOMALY_Z_THRESHOLD:
            INPUT_ANOMALIES_TOTAL.labels(
                feature=name, anomaly_type="out_of_range"
            ).inc()
            # Don't reject — just flag. The model can still produce a
            # prediction; the user's job is to investigate the spike.

    # ----- Inference (timed separately from total request) ----- #
    inf_start = time.perf_counter()
    pred_class = int(state.model.predict(feature_vec_scaled := scaled.reshape(1, -1))[0])
    proba = state.model.predict_proba(feature_vec_scaled)[0]
    inf_elapsed = time.perf_counter() - inf_start

    INFERENCE_DURATION_SECONDS.observe(inf_elapsed)

    # ----- Update prediction metrics ----- #
    class_name = state.reference.class_names[pred_class]
    PREDICTIONS_TOTAL.labels(predicted_class=class_name).inc()
    confidence = float(proba.max())
    PREDICTION_CONFIDENCE.observe(confidence)

    # ----- Feed rolling window for drift PSI ----- #
    state.window.add(scaled_dict, pred_class)

    return PredictResponse(
        predicted_class=pred_class,
        predicted_class_name=class_name,
        probabilities={
            state.reference.class_names[i]: float(p) for i, p in enumerate(proba)
        },
        confidence=confidence,
        inference_ms=inf_elapsed * 1000,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensures error responses still have proper status codes for the
    middleware to record."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
