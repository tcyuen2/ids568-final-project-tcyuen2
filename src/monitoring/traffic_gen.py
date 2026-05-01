"""
Synthetic traffic generator for the Iris classifier service.

Supports three scenarios via --scenario:

  normal   - Sample from the original Iris distribution. Class proportions
             match training (~33/33/33), feature drift PSI stays < 0.1.
             Use this to populate the dashboard with healthy baseline data.

  drift    - Gradually shifts the input distribution over time: petal
             measurements stretch toward the virginica end of the range.
             Per-feature PSI rises into the "moderate" (>0.1) and then
             "significant" (>0.25) bands; class distribution skews
             toward virginica. This is the scenario the dashboard's
             drift panel should make obvious.

  anomalies - Standard traffic plus a 5% rate of malformed inputs:
              negative values, zeroes where impossible, extreme outliers,
              and occasional NaNs. Drives the input_anomalies_total
              counter and demonstrates the alerting story.

Run alongside the service:
    python -m src.monitoring.traffic_gen --scenario drift --rate 5 --duration 300
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import json as _json
from dataclasses import dataclass

import numpy as np
import requests
from sklearn.datasets import load_iris

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IrisDistribution:
    """Per-class mean / std for sampling synthetic Iris-like inputs.

    Fitted on the original sklearn Iris dataset, used as a generative
    model so we can synthesize traffic that looks like real Iris data
    without simply replaying the training set (which would make drift
    metrics trivially zero).
    """

    means: np.ndarray  # shape (3, 4): per-class means in raw cm units
    stds: np.ndarray   # shape (3, 4): per-class stds

    @classmethod
    def from_iris(cls) -> "IrisDistribution":
        iris = load_iris()
        X, y = iris.data, iris.target
        means = np.stack([X[y == c].mean(axis=0) for c in range(3)])
        stds = np.stack([X[y == c].std(axis=0) for c in range(3)])
        return cls(means=means, stds=stds)

    def sample(self, class_id: int) -> dict[str, float]:
        m = self.means[class_id]
        s = self.stds[class_id]
        vals = np.random.normal(loc=m, scale=s)
        # Clip to physically plausible range (no negative measurements)
        vals = np.clip(vals, 0.05, None)
        return {
            "sepal_length": float(vals[0]),
            "sepal_width": float(vals[1]),
            "petal_length": float(vals[2]),
            "petal_width": float(vals[3]),
        }


def sample_normal(dist: IrisDistribution) -> dict[str, float]:
    """Pick a class uniformly, sample a synthetic example."""
    class_id = random.randint(0, 2)
    return dist.sample(class_id)


def sample_with_drift(
    dist: IrisDistribution, drift_progress: float
) -> dict[str, float]:
    """Drifted sample: petal measurements scale up over time.

    `drift_progress` runs 0.0 -> 1.0 over the run. At progress=1.0,
    petal length and width have been multiplicatively scaled by 1.5x —
    enough to push PSI into the >0.25 "significant drift" band.
    """
    sample = sample_normal(dist)
    scale = 1.0 + 0.5 * drift_progress
    sample["petal_length"] *= scale
    sample["petal_width"] *= scale
    return sample


def sample_anomaly() -> dict[str, float]:
    """Generate a deliberately malformed input. The handler logs an
    integrity-anomaly counter for these (and a few will be rejected
    outright on NaN)."""
    kind = random.choice(["extreme", "negative", "zero", "nan"])
    if kind == "extreme":
        return {
            "sepal_length": random.uniform(40, 60),
            "sepal_width": random.uniform(20, 40),
            "petal_length": random.uniform(30, 50),
            "petal_width": random.uniform(15, 25),
        }
    if kind == "negative":
        return {
            "sepal_length": -random.uniform(1, 5),
            "sepal_width": -random.uniform(1, 3),
            "petal_length": random.uniform(1, 5),
            "petal_width": random.uniform(0.1, 2),
        }
    if kind == "zero":
        return {
            "sepal_length": 0.0,
            "sepal_width": 0.0,
            "petal_length": random.uniform(1, 5),
            "petal_width": random.uniform(0.1, 2),
        }
    # NaN
    return {
        "sepal_length": float("nan"),
        "sepal_width": random.uniform(2, 4),
        "petal_length": random.uniform(1, 5),
        "petal_width": random.uniform(0.1, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="Predict endpoint URL.",
    )
    parser.add_argument(
        "--scenario",
        choices=["normal", "drift", "anomalies"],
        default="normal",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Target requests per second.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Total run time in seconds (0 = run forever).",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Fraction of requests that are anomalous in 'anomalies' scenario.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    dist = IrisDistribution.from_iris()
    delay = 1.0 / args.rate if args.rate > 0 else 0
    start = time.time()
    sent = 0
    errors = 0
    rejected = 0  # 4xx responses (e.g., NaN inputs we reject)

    logger.info(
        "Generating %s traffic at %.1f rps for %.0fs (target=%s)",
        args.scenario,
        args.rate,
        args.duration,
        args.url,
    )

    try:
        while True:
            elapsed = time.time() - start
            if args.duration > 0 and elapsed >= args.duration:
                break

            if args.scenario == "normal":
                payload = sample_normal(dist)
            elif args.scenario == "drift":
                progress = min(1.0, elapsed / max(args.duration, 1))
                payload = sample_with_drift(dist, progress)
            else:  # anomalies
                if random.random() < args.anomaly_rate:
                    payload = sample_anomaly()
                else:
                    payload = sample_normal(dist)

            try:
                body = _json.dumps(payload, allow_nan=True)
                r = requests.post(
                    args.url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    timeout=2.0,
                )
                if 400 <= r.status_code < 500:
                    rejected += 1
                elif r.status_code >= 500:
                    errors += 1
            except requests.RequestException as e:
                errors += 1
                if errors <= 3:
                    logger.warning("Request error: %s", e)

            sent += 1
            if sent % 50 == 0:
                logger.info(
                    "sent=%d ok=%d rejected_4xx=%d errors=%d elapsed=%.0fs",
                    sent,
                    sent - rejected - errors,
                    rejected,
                    errors,
                    elapsed,
                )
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    logger.info(
        "Done. sent=%d ok=%d rejected_4xx=%d errors=%d",
        sent,
        sent - rejected - errors,
        rejected,
        errors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
