"""
Generate the Component 3 lineage diagram.

Produces docs/lineage-diagram.png — a horizontal flow diagram showing
how data, code, and artifacts move from M3 training through Final
Project deployment and monitoring.

Reproducibility: this script is kept in the repo (not just the PNG)
so the diagram can be regenerated if the system structure changes.
Run from repo root:
    python docs/generate_lineage_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Layout: 5 columns x variable rows, left to right.
# Each column is a "stage"; each box within a column is an artifact or step.
# ---------------------------------------------------------------------------

STAGE_COLORS = {
    "data": "#dbeafe",        # blue-100
    "training": "#fce7f3",    # pink-100
    "registry": "#fef3c7",    # yellow-100
    "serving": "#dcfce7",     # green-100
    "monitoring": "#e9d5ff",  # purple-100
    "governance": "#fee2e2",  # red-100
}

STAGE_BORDER = {
    "data": "#1e40af",
    "training": "#9d174d",
    "registry": "#92400e",
    "serving": "#166534",
    "monitoring": "#6b21a8",
    "governance": "#991b1b",
}


def draw_box(ax, x, y, w, h, stage, lines, fontsize=9, fontweight="normal"):
    """Draw a rounded rectangle with multi-line text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.4,
        facecolor=STAGE_COLORS[stage],
        edgecolor=STAGE_BORDER[stage],
    )
    ax.add_patch(box)
    text = "\n".join(lines)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color="#111827",
        wrap=True,
    )


def draw_arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0.0, 0.08)):
    """Draw an arrow with an optional label."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#374151",
            lw=1.4,
            mutation_scale=14,
        ),
    )
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="#4b5563",
            style="italic",
        )


def main():
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(
        8.0, 8.55,
        "Iris Classifier — Lineage from Training to Production Monitoring",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        8.0, 8.15,
        "Milestone 3 (training) → Final Project Components 1–3 (serving, monitoring, governance)",
        ha="center",
        fontsize=10,
        color="#6b7280",
        style="italic",
    )

    # ---- Column 1: DATA (x ≈ 0.3–2.7) ----
    draw_box(ax, 0.3, 5.5, 2.4, 1.4, "data",
             ["Iris dataset", "(Fisher 1936)", "150 samples × 4 features",
              "via sklearn.datasets"], fontweight="bold")
    draw_box(ax, 0.3, 3.7, 2.4, 1.4, "data",
             ["Preprocessing", "+ 2 interactions", "StandardScaler",
              "(M3 train.py)"])
    draw_box(ax, 0.3, 1.9, 2.4, 1.4, "data",
             ["Stratified split", "120 train / 30 test", "data SHA-256 hashed",
              "(M3 lineage_report.md)"])

    # ---- Column 2: TRAINING (x ≈ 3.4–5.8) ----
    draw_box(ax, 3.4, 5.5, 2.4, 1.4, "training",
             ["RandomForest training", "5 hyperparameter configs",
              "(M3 run_experiments.py)"], fontweight="bold")
    draw_box(ax, 3.4, 3.7, 2.4, 1.4, "training",
             ["Quality gates", "acc ≥ 0.90  F1 ≥ 0.85", "AUC ≥ 0.90",
              "(M3 model_validation.py)"])
    draw_box(ax, 3.4, 1.9, 2.4, 1.4, "training",
             ["MLflow tracking", "metrics + params", "+ artifact hashes"])

    # ---- Column 3: REGISTRY / A-B (x ≈ 6.5–9.0) ----
    draw_box(ax, 6.5, 5.5, 2.5, 1.4, "registry",
             ["MLflow run", "0ab6a6fa... (B)", "baseline_run", "n=50, depth=5"], fontweight="bold")
    draw_box(ax, 6.5, 3.7, 2.5, 1.4, "registry",
             ["MLflow run", "6e01be84... (A)", "high_min_split", "n=100, depth=10"])
    draw_box(ax, 6.5, 1.9, 2.5, 1.4, "registry",
             ["A/B test (C2)", "n=11,430 total", "α=0.05  margin=3%",
              "decision: SHIP B"])

    # ---- Column 4: SERVING (x ≈ 9.7–12.2) ----
    draw_box(ax, 9.7, 5.5, 2.5, 1.4, "serving",
             ["Bundled artifact", "models/model.pkl", "+ scaler.pkl",
              "active variant: B"], fontweight="bold")
    draw_box(ax, 9.7, 3.7, 2.5, 1.4, "serving",
             ["FastAPI service", "/predict /metrics /health",
              "Pydantic validation", "Docker container"])
    draw_box(ax, 9.7, 1.9, 2.5, 1.4, "serving",
             ["Live integrity", "NaN/Inf rejection",
              "|z|>5 anomaly check"])

    # ---- Column 5: MONITORING (x ≈ 12.9–15.4) ----
    draw_box(ax, 12.9, 5.5, 2.5, 1.4, "monitoring",
             ["Prometheus", "5s scrape", "12 metrics emitted"], fontweight="bold")
    draw_box(ax, 12.9, 3.7, 2.5, 1.4, "monitoring",
             ["Grafana dashboard", "6 panels", "RED + drift + integrity",
              "thresholds: 0.1 / 0.25"])
    draw_box(ax, 12.9, 1.9, 2.5, 1.4, "monitoring",
             ["Drift detection (C4)", "PSI vs. reference",
              "feature + output drift"])

    # ---- Bottom row: GOVERNANCE (full width, x = 0.3 to 15.4) ----
    draw_box(ax, 0.3, 0.2, 15.1, 1.2, "governance",
             ["Governance & Audit (Component 3)",
              "Model card  •  Risk register  •  Audit trail (logs/audit-trail.json)  •  This lineage diagram",
              "Every event above is recorded as an append-only audit-trail entry."],
             fontsize=10, fontweight="bold")

    # ---- Arrows between columns (horizontal flow) ----
    # data -> training (top row)
    draw_arrow(ax, 2.7, 6.2, 3.4, 6.2)
    # data -> training (mid row)
    draw_arrow(ax, 2.7, 4.4, 3.4, 4.4)
    # data -> training (bottom row)
    draw_arrow(ax, 2.7, 2.6, 3.4, 2.6)
    # training -> registry
    draw_arrow(ax, 5.8, 6.2, 6.5, 6.2)
    draw_arrow(ax, 5.8, 4.4, 6.5, 4.4)
    draw_arrow(ax, 5.8, 2.6, 6.5, 2.6)
    # registry -> serving
    draw_arrow(ax, 9.0, 6.2, 9.7, 6.2, label="SHIP_B promotion", label_offset=(0.0, 0.18))
    draw_arrow(ax, 9.0, 4.4, 9.7, 4.4, label="(previous)", label_offset=(0.0, 0.16))
    draw_arrow(ax, 9.0, 2.6, 9.7, 2.6)
    # serving -> monitoring
    draw_arrow(ax, 12.2, 6.2, 12.9, 6.2)
    draw_arrow(ax, 12.2, 4.4, 12.9, 4.4)
    draw_arrow(ax, 12.2, 2.6, 12.9, 2.6)

    # Vertical arrows from each major stage down to governance
    for x in [1.5, 4.6, 7.75, 10.95, 14.15]:
        draw_arrow(ax, x, 1.85, x, 1.5)

    # ---- Legend ----
    legend_handles = [
        mpatches.Patch(color=STAGE_COLORS["data"], label="Data (M3)"),
        mpatches.Patch(color=STAGE_COLORS["training"], label="Training (M3)"),
        mpatches.Patch(color=STAGE_COLORS["registry"], label="Registry / Experiments"),
        mpatches.Patch(color=STAGE_COLORS["serving"], label="Serving (C1)"),
        mpatches.Patch(color=STAGE_COLORS["monitoring"], label="Monitoring (C1, C4)"),
        mpatches.Patch(color=STAGE_COLORS["governance"], label="Governance (C3)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              ncol=6, frameon=False, fontsize=9)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "lineage-diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
