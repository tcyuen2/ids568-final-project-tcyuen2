"""
Visualizations for the C4 drift analysis.

Produces four PNGs in visualizations/drift/:
  1. psi-evolution.png — PSI per feature across the 3 production windows
     with 0.10 / 0.25 threshold lines (mirrors the C1 dashboard panel).
  2. feature-distributions.png — per-feature histograms overlaying
     reference vs. each production window.
  3. accuracy-impact.png — bar chart: predicted accuracy per window vs.
     reference baseline. The headline impact-analysis chart.
  4. class-distribution-shift.png — stacked bar showing how the predicted
     class distribution evolves across windows.

Run:
    python -m src.drift.visualize

Inputs:
    src/drift/results.json (from detect.py)
    src/drift/data/*.csv   (raw window data for distribution overlays)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "src" / "drift" / "results.json"
DATA_DIR = REPO_ROOT / "src" / "drift" / "data"
OUTPUT_DIR = REPO_ROOT / "visualizations" / "drift"

# Keep the Week 1 / Week 2 / Week 3 colors consistent across all four
# charts so the eye can track the same window across plots.
WINDOW_COLORS = {
    "reference":         "#6b7280",  # gray (baseline)
    "production_week1":  "#10b981",  # green (healthy)
    "production_week2":  "#f59e0b",  # amber (moderate drift)
    "production_week3":  "#ef4444",  # red (significant drift)
}
WINDOW_LABELS = {
    "reference":        "Reference (training distribution)",
    "production_week1": "Week 1 — clean traffic",
    "production_week2": "Week 2 — moderate drift (petal × 1.2)",
    "production_week3": "Week 3 — significant drift + integrity issues",
}
PRODUCTION_WINDOWS = ["production_week1", "production_week2", "production_week3"]


def plot_psi_evolution(results: dict, out_path: Path) -> None:
    """PSI per feature across the 3 production windows with threshold bands."""
    feature_names = list(results["windows"]["reference"]["feature_drift"].keys())
    windows = PRODUCTION_WINDOWS

    fig, ax = plt.subplots(figsize=(11, 6))

    # PSI threshold bands — 0.0–0.1 stable, 0.1–0.25 moderate, >0.25 significant.
    ax.axhspan(0,    0.10, color="#dcfce7", alpha=0.6, label="Stable (PSI < 0.1)")
    ax.axhspan(0.10, 0.25, color="#fef3c7", alpha=0.6, label="Moderate (0.1–0.25)")
    ax.axhspan(0.25, 5.0,  color="#fee2e2", alpha=0.45, label="Significant (PSI > 0.25)")

    # One line per feature, marker per window
    week_labels = ["Week 1", "Week 2", "Week 3"]
    x = np.arange(len(week_labels))
    for fname in feature_names:
        psi_values = [results["windows"][w]["feature_drift"][fname]["psi"] for w in windows]
        ax.plot(x, psi_values, marker="o", linewidth=2, markersize=8, label=fname)

    ax.set_xticks(x)
    ax.set_xticklabels(week_labels, fontsize=11)
    ax.set_ylabel("PSI vs. reference distribution", fontsize=11)
    ax.set_title("Per-feature PSI evolution across production windows\n"
                 "(thresholds match the live Component 1 dashboard)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))  # don't let band labels squish if drift is small
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def plot_feature_distributions(results: dict, out_path: Path) -> None:
    """Reference vs. each production window, one panel per raw feature."""
    raw_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Pre-load each window's raw values once
    window_dfs = {w: pd.read_csv(DATA_DIR / f"{w}.csv") for w in WINDOW_COLORS}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, fname in zip(axes.flat, raw_features):
        for wname, df in window_dfs.items():
            values = df[fname].dropna().values
            ax.hist(values, bins=30, alpha=0.45, label=WINDOW_LABELS[wname],
                    color=WINDOW_COLORS[wname], edgecolor="white", linewidth=0.5)
        ax.set_title(fname, fontsize=11, fontweight="bold")
        ax.set_xlabel("value (cm)")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)

    # Single legend for all 4 panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=9, framealpha=0.95)
    fig.suptitle("Raw feature distributions: reference vs. production windows",
                 fontsize=13, fontweight="bold", y=1.07)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def plot_accuracy_impact(results: dict, out_path: Path) -> None:
    """Bar chart of accuracy per window — the headline impact figure."""
    ref_acc = results["reference_accuracy"]
    windows_to_plot = ["reference"] + PRODUCTION_WINDOWS
    accuracies = [results["windows"][w]["impact"]["accuracy"] for w in windows_to_plot]
    confidences = [results["windows"][w]["impact"]["mean_confidence"] for w in windows_to_plot]
    colors = [WINDOW_COLORS[w] for w in windows_to_plot]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ----- Left panel: accuracy with reference dashed line -----
    bars = ax1.bar(range(len(windows_to_plot)), accuracies, color=colors,
                   edgecolor="black", linewidth=0.7)
    ax1.axhline(ref_acc, color="#6b7280", linestyle="--", linewidth=1.4,
                label=f"Reference baseline = {ref_acc:.3f}")
    ax1.set_xticks(range(len(windows_to_plot)))
    ax1.set_xticklabels(["Reference", "Week 1", "Week 2", "Week 3"], fontsize=10)
    ax1.set_ylabel("Model accuracy on window", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Accuracy degradation as drift increases", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower left")
    ax1.grid(axis="y", alpha=0.3)
    # Value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f"{acc:.3f}", ha="center", fontsize=10, fontweight="bold")

    # ----- Right panel: mean confidence (leading indicator) -----
    bars2 = ax2.bar(range(len(windows_to_plot)), confidences, color=colors,
                    edgecolor="black", linewidth=0.7)
    ax2.set_xticks(range(len(windows_to_plot)))
    ax2.set_xticklabels(["Reference", "Week 1", "Week 2", "Week 3"], fontsize=10)
    ax2.set_ylabel("Mean prediction confidence", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Mean prediction confidence (leading indicator)",
                  fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, conf in zip(bars2, confidences):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f"{conf:.3f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Impact analysis: drift → accuracy & confidence",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def plot_class_distribution_shift(results: dict, out_path: Path) -> None:
    """Stacked bar of predicted class proportions per window — output drift."""
    windows = ["reference"] + PRODUCTION_WINDOWS
    class_names = ["setosa", "versicolor", "virginica"]
    pcts_per_class = {c: [] for c in class_names}
    for w in windows:
        actual = results["windows"][w]["output_drift"]["actual_class_pcts"]
        for i, c in enumerate(class_names):
            pcts_per_class[c].append(actual[i])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(windows))
    bottom = np.zeros(len(windows))
    class_colors = ["#22c55e", "#eab308", "#3b82f6"]  # green / yellow / blue
    for c, color in zip(class_names, class_colors):
        ax.bar(x, pcts_per_class[c], bottom=bottom, label=c, color=color,
               edgecolor="white", linewidth=1.2)
        # Percentage labels in the middle of each segment
        for i, val in enumerate(pcts_per_class[c]):
            if val > 0.04:  # don't label tiny slivers
                ax.text(i, bottom[i] + val / 2, f"{val:.0%}",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white")
        bottom += np.array(pcts_per_class[c])

    # Reference horizontal line at 0.333 (the training distribution per class)
    ax.axhline(0.333, color="#6b7280", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.667, color="#6b7280", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(len(windows) - 0.5, 0.333, "33%", fontsize=8, color="#6b7280",
            va="bottom", ha="right", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Reference", "Week 1", "Week 2", "Week 3"], fontsize=10)
    ax.set_ylabel("Predicted class proportion", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Output drift: predicted class distribution by window\n"
                 "(reference is class-balanced at 33% / 33% / 33%)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    logger.info("Generating C4 drift visualizations...")
    plot_psi_evolution(results,           OUTPUT_DIR / "psi-evolution.png")
    plot_feature_distributions(results,   OUTPUT_DIR / "feature-distributions.png")
    plot_accuracy_impact(results,         OUTPUT_DIR / "accuracy-impact.png")
    plot_class_distribution_shift(results, OUTPUT_DIR / "class-distribution-shift.png")
    logger.info("Done. 4 charts in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
