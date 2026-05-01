"""
Generate reference + production windows for C4 drift analysis.

Produces four CSVs in src/drift/data/:
    reference.csv          ← stratified sample of M3 training data + true labels
    production_week1.csv   ← clean traffic (no drift, healthy baseline)
    production_week2.csv   ← moderate drift (petal scaling 1.2x)
    production_week3.csv   ← significant drift (petal scaling 1.5x + integrity issues)

Why fixed windows rather than streaming:
    - C1 already covers live streaming drift in Prometheus + Grafana.
    - C4 is the *offline* counterpart: snapshot windows let us run rich
      statistical tests (PSI, KS, Wasserstein) and confirm what C1 flagged.
    - The 3-window structure mirrors how a real production team would
      slice traffic: "this week vs. baseline" rather than continuous.

Reproducibility: seeded numpy RNG. Same seed = identical windows.

Builds on Milestone 4: this script adapts the per-class normal sampling
pattern from M4's generate_data.py to produce drift-injected windows
rather than uniform synthetic data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "src" / "drift" / "data"

# Raw feature names — matches M3 ordering and the FastAPI /predict contract
RAW_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def fit_class_distributions() -> tuple[np.ndarray, np.ndarray]:
    """Fit per-class normal distributions to the original Iris data.

    Returns (means, stds) each of shape (3, 4): per (class, feature).
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    means = np.stack([X[y == c].mean(axis=0) for c in range(3)])
    stds  = np.stack([X[y == c].std(axis=0)  for c in range(3)])
    return means, stds


def sample_window(
    n: int,
    means: np.ndarray,
    stds: np.ndarray,
    rng: np.random.Generator,
    petal_scale: float = 1.0,
    integrity_anomaly_rate: float = 0.0,
) -> pd.DataFrame:
    """Sample a window of n labeled requests.

    Args:
        n: number of samples in the window.
        means, stds: per-class distribution params from fit_class_distributions.
        rng: numpy random generator (for reproducibility).
        petal_scale: multiplier on petal_length and petal_width AFTER sampling.
            1.0 = no drift; 1.2 = moderate; 1.5 = significant.
        integrity_anomaly_rate: fraction of rows to corrupt with NaN/extreme
            values. Simulates upstream data quality issues.

    Returns: DataFrame with the 4 raw features and the true class label.
    """
    # Uniform class draws: each window is class-balanced so drift effects
    # don't confound with class-imbalance effects.
    labels = rng.integers(0, 3, size=n)
    samples = np.zeros((n, 4))
    for i, c in enumerate(labels):
        samples[i] = rng.normal(means[c], stds[c])
    samples = np.clip(samples, 0.05, None)

    # Inject drift: scale petal measurements only (mimics a real-world
    # scenario where one upstream sensor's calibration shifted but others
    # are stable).
    if petal_scale != 1.0:
        samples[:, 2] *= petal_scale  # petal_length
        samples[:, 3] *= petal_scale  # petal_width

    # Inject integrity anomalies (NaN or extreme values).
    if integrity_anomaly_rate > 0:
        n_anomalies = int(n * integrity_anomaly_rate)
        anomaly_idx = rng.choice(n, n_anomalies, replace=False)
        for idx in anomaly_idx:
            # Half NaN, half wildly out of range
            if rng.random() < 0.5:
                col = rng.integers(0, 4)
                samples[idx, col] = np.nan
            else:
                col = rng.integers(0, 4)
                samples[idx, col] = rng.uniform(40, 60)

    df = pd.DataFrame(samples, columns=RAW_FEATURES)
    df["target"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000,
                       help="Samples per window (default 2000 — large enough for KS test power, small enough to chart fast)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    means, stds = fit_class_distributions()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    windows = {
        # Reference: clean Iris-distributed traffic, no drift, no anomalies.
        # This is what the model was trained on and what we compare against.
        "reference":         (1.0, 0.0),
        # Week 1: normal production traffic. Same distribution as reference
        # but a fresh sample, so trivial PSI from sampling noise expected.
        "production_week1":  (1.0, 0.0),
        # Week 2: moderate drift. Petal measurements 20% larger.
        # Simulates: upstream sensor calibration drift on petal channel.
        "production_week2":  (1.2, 0.0),
        # Week 3: significant drift + integrity issues. Petals 50% larger
        # and 5% of rows have NaN or extreme values. Simulates a partial
        # upstream pipeline failure.
        "production_week3":  (1.5, 0.05),
    }

    for name, (petal_scale, anom_rate) in windows.items():
        df = sample_window(args.n, means, stds, rng,
                           petal_scale=petal_scale,
                           integrity_anomaly_rate=anom_rate)
        out_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        # Quick summary so the operator can see what was produced
        nan_count = df[RAW_FEATURES].isna().sum().sum()
        extreme_count = (df[RAW_FEATURES].abs() > 30).sum().sum()
        print(f"  {name:20s}  n={len(df)}  petal_scale={petal_scale}  "
              f"anom_rate={anom_rate}  nans={nan_count}  extremes={extreme_count}")

    print(f"\nWrote 4 windows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
