"""
Generate reference distribution statistics from training data.

These stats are loaded by the serving service to compute live PSI
(Population Stability Index) per feature against the training distribution.

This script mirrors the preprocessing in the M3 train_pipeline:
  - Load iris
  - Add 2 interaction features (sepal_l*sepal_w, petal_l*petal_w)
  - Apply StandardScaler (loaded from the bundled scaler.pkl)

Run once after model artifacts are bundled. The output file
(models/reference_stats.json) is committed to the repo so the service
is self-contained and reproducible.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris

# Feature names match the order produced by M3 preprocessing
FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "sepal_l_x_sepal_w",  # interaction term 1
    "petal_l_x_petal_w",  # interaction term 2
]

# Number of bins for PSI calculation. 10 is the standard convention from
# credit-risk literature (Siddiqi). Fewer bins = less sensitive to noise
# but coarser drift detection; 10 is a well-balanced default.
N_PSI_BINS = 10


def main():
    repo_root = Path(__file__).resolve().parents[2]
    scaler_path = repo_root / "models" / "scaler.pkl"
    output_path = repo_root / "models" / "reference_stats.json"

    # Load the scaler that was fit during M3 training
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Recreate the M3 training feature matrix
    iris = load_iris()
    X = iris.data
    X_with_interactions = np.c_[X, X[:, 0] * X[:, 1], X[:, 2] * X[:, 3]]
    X_scaled = scaler.transform(X_with_interactions)

    assert X_scaled.shape[1] == len(FEATURE_NAMES), (
        f"Feature count mismatch: {X_scaled.shape[1]} vs {len(FEATURE_NAMES)}"
    )

    reference = {
        "feature_names": FEATURE_NAMES,
        "n_samples": int(X_scaled.shape[0]),
        "psi_n_bins": N_PSI_BINS,
        "features": {},
    }

    # Per-feature statistics needed at serving time:
    #   - mean / std: for live mean/std gauges and rough sanity checks
    #   - min / max: helps frame "out-of-range" anomaly detection
    #   - bin_edges: PSI bins fixed from the training distribution; live
    #     data gets binned into these same edges so distributions are
    #     directly comparable.
    #   - bin_pcts: training distribution proportions per bin (the "expected"
    #     side of the PSI formula).
    for i, name in enumerate(FEATURE_NAMES):
        col = X_scaled[:, i]

        # Use quantile-based bin edges so each training bin has roughly
        # equal mass. This is a more robust PSI variant than fixed-width
        # bins, especially for skewed features.
        quantiles = np.linspace(0, 1, N_PSI_BINS + 1)
        bin_edges = np.quantile(col, quantiles)
        # Ensure strictly increasing edges (can fail when many ties exist)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            # Degenerate column; fall back to fixed-width
            bin_edges = np.linspace(col.min(), col.max(), N_PSI_BINS + 1)

        # Replace outermost edges with -inf / +inf so live values outside
        # the training range still get binned (avoids divide-by-zero in PSI).
        bin_edges_with_inf = bin_edges.copy()
        bin_edges_with_inf[0] = -np.inf
        bin_edges_with_inf[-1] = np.inf

        # Training distribution proportions per bin
        counts, _ = np.histogram(col, bins=bin_edges_with_inf)
        bin_pcts = counts / counts.sum()

        # Smooth zeros (PSI of log(0) is undefined). Standard fix.
        bin_pcts = np.where(bin_pcts == 0, 1e-6, bin_pcts)

        reference["features"][name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "bin_edges": bin_edges_with_inf.tolist(),
            "bin_pcts": bin_pcts.tolist(),
        }

    # Also save the training class proportions for output drift comparison.
    # For stratified Iris this is uniform (~33/33/33).
    y = iris.target
    class_pcts = {
        int(c): float((y == c).sum() / len(y)) for c in np.unique(y)
    }
    reference["class_pcts"] = class_pcts
    reference["class_names"] = iris.target_names.tolist()

    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"Wrote reference stats to {output_path}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Samples used: {reference['n_samples']}")
    print(f"  PSI bins per feature: {N_PSI_BINS}")


if __name__ == "__main__":
    main()
