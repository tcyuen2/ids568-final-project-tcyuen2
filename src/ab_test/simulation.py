"""
A/B simulation for the Iris classifier.

Implements the experiment design specified in
docs/experiment-specification.md:
  - Variant A (control): high_min_split (M3 run 6e01be84...)
  - Variant B (challenger): baseline_run (M3 run 0ab6a6fa...)
  - Hash-based 50/50 randomization
  - Synthetic traffic from per-class normal distributions fitted to Iris
  - Output: per-request results to a JSON file consumed by analyze.py

Usage:
    python -m src.ab_test.simulation
    python -m src.ab_test.simulation --n-total 11430 --rate 5 --seed 42

The simulation does NOT run against the live FastAPI service. It applies
the same preprocessing and the same models that the service would use,
but in-process — much faster and deterministic. The instrumentation
patterns are identical (so we could easily switch to live HTTP if needed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
import uuid
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS_DIR = REPO_ROOT / "models" / "variants"
RESULTS_PATH = REPO_ROOT / "src" / "ab_test" / "results.json"

# Lineage: which M3 MLflow run produced each variant.
VARIANT_LINEAGE = {
    "A": {"name": "high_min_split", "m3_run_id": "6e01be84da0e47d68b1fc5caa9749e40"},
    "B": {"name": "baseline_run",   "m3_run_id": "0ab6a6fa80f949d5b917115218e126c4"},
}


def assign_variant(request_id: str) -> str:
    """Hash-based bucketing: stable 50/50 split.

    Same logic as the experiment specification — we hash the request_id
    (a UUID), take it mod 100, and bucket by threshold. Stable, so the
    same request always lands in the same variant; independent across
    requests.
    """
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return "B" if (h % 100) < 50 else "A"


def load_variant(name: str) -> tuple[object, object]:
    """Load model and scaler pickles for a named variant."""
    with open(VARIANTS_DIR / f"{name}_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(VARIANTS_DIR / f"{name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def preprocess(raw: np.ndarray, scaler) -> np.ndarray:
    """M3 preprocessing pipeline: add interaction features, scale.

    raw: shape (N, 4) — sepal_length, sepal_width, petal_length, petal_width.
    """
    interactions = np.c_[raw, raw[:, 0] * raw[:, 1], raw[:, 2] * raw[:, 3]]
    return scaler.transform(interactions)


def synthesize_traffic(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate n labeled synthetic Iris-like requests.

    Returns (raw_features [n,4], true_labels [n]). Sampling is per-class
    normal with parameters fit to the original Iris dataset; class is
    drawn uniformly so guardrails (e.g., class skew) can be assessed
    against a known-balanced ground truth.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    means = np.stack([X[y == c].mean(axis=0) for c in range(3)])
    stds  = np.stack([X[y == c].std(axis=0)  for c in range(3)])

    labels = rng.integers(0, 3, size=n)
    samples = np.zeros((n, 4))
    for i, c in enumerate(labels):
        samples[i] = rng.normal(means[c], stds[c])
    # Clip to physically plausible (no negative cm) — same as traffic_gen.py
    samples = np.clip(samples, 0.05, None)
    return samples, labels


def run_simulation(n_total: int, seed: int, rate: float | None = None) -> dict:
    """Run the full A/B simulation end to end.

    Args:
        n_total: total number of requests across both arms.
        seed: numpy RNG seed for reproducibility.
        rate: if set, simulate at most `rate` requests per second
            (sleeps between requests). Use None for batch (fastest).

    Returns: a dict suitable for JSON serialization with per-variant
        outcomes and the per-request raw log.
    """
    rng = np.random.default_rng(seed)

    # Pre-load both variants once.
    model_A, scaler_A = load_variant("high_min_split")
    model_B, scaler_B = load_variant("baseline_run")
    logger.info("Loaded variants A=high_min_split B=baseline_run")

    # Pre-generate all synthetic traffic and ground truth — keeps
    # randomness deterministic regardless of timing.
    raw, labels = synthesize_traffic(n_total, rng)
    logger.info("Synthesized n=%d requests", n_total)

    # Per-variant counters (the primary metric is accuracy).
    counts = {"A": {"correct": 0, "total": 0}, "B": {"correct": 0, "total": 0}}
    # We record per-request outcomes so analyze.py can compute CIs and
    # any secondary stats. Memory-light: a flat list of small dicts.
    log: list[dict] = []

    # Pre-cache scaled features for each variant. Scaling is the same
    # across both variants in this experiment because both M3 runs
    # used the same StandardScaler params, but in general the scaler
    # could differ between variants — keeping them separate is safer.
    scaled_A = preprocess(raw, scaler_A)
    scaled_B = preprocess(raw, scaler_B)

    delay = (1.0 / rate) if rate else 0.0
    start = time.time()

    for i in range(n_total):
        # numpy RNG is bounded by int64. Build a 128-bit UUID from two
        # 64-bit chunks so the request_id stream is fully reproducible
        # given the seed (uuid.uuid4() would use OS randomness instead).
        hi = int(rng.integers(0, 2**63 - 1))
        lo = int(rng.integers(0, 2**63 - 1))
        request_id = str(uuid.UUID(int=(hi << 64) | lo))
        variant = assign_variant(request_id)

        if variant == "A":
            pred = int(model_A.predict(scaled_A[i:i+1])[0])
        else:
            pred = int(model_B.predict(scaled_B[i:i+1])[0])

        true = int(labels[i])
        correct = (pred == true)
        counts[variant]["total"] += 1
        counts[variant]["correct"] += int(correct)

        log.append({
            "i": i,
            "request_id": request_id,
            "variant": variant,
            "predicted": pred,
            "true": true,
            "correct": correct,
        })

        if delay:
            time.sleep(delay)
        if (i + 1) % 1000 == 0:
            logger.info("progress: i=%d/%d  A_n=%d  B_n=%d", i + 1, n_total,
                        counts["A"]["total"], counts["B"]["total"])

    elapsed = time.time() - start

    # Verify the split landed near 50/50. If it's wildly off, something
    # is wrong with the hash function or the assignment logic.
    split_a = counts["A"]["total"] / n_total
    if abs(split_a - 0.5) > 0.05:
        logger.warning("Split is %.3f (A) / %.3f (B) — outside ±5%% of 50/50",
                       split_a, 1 - split_a)
    else:
        logger.info("Split confirmed near 50/50: A=%.3f B=%.3f", split_a, 1 - split_a)

    # Final summary.
    acc_A = counts["A"]["correct"] / counts["A"]["total"] if counts["A"]["total"] else 0
    acc_B = counts["B"]["correct"] / counts["B"]["total"] if counts["B"]["total"] else 0
    logger.info("Done. acc_A=%.4f (n=%d)  acc_B=%.4f (n=%d)  elapsed=%.1fs",
                acc_A, counts["A"]["total"], acc_B, counts["B"]["total"], elapsed)

    return {
        "config": {
            "n_total": n_total,
            "seed": seed,
            "rate_rps": rate,
            "elapsed_seconds": elapsed,
            "variants": VARIANT_LINEAGE,
        },
        "summary": {
            "A": {**counts["A"], "accuracy": acc_A},
            "B": {**counts["B"], "accuracy": acc_B},
        },
        "log": log,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-total", type=int, default=11430,
                   help="Total requests across both arms (default: 11430, the sample size from the spec)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rate", type=float, default=None,
                   help="Optional throttling in req/s. Default: as fast as possible (batch).")
    p.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = p.parse_args()

    results = run_simulation(args.n_total, args.seed, args.rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
