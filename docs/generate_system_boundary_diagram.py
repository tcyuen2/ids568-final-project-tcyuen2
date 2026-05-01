"""
Generate the Component 5 system boundary diagram.

The rubric example shows "retriever → LLM → tool use → final output" — a
shape that does not match this system. This diagram instead depicts the
*actual* boundary of the deployed Iris classifier service: how data
enters, where it's transformed, what produces predictions, what watches
for problems, and where governance attaches.

The diagram explicitly marks what is NOT in this system (LLM, vector
store, agentic tools, external API integrations) so the boundary is
unambiguous to a reviewer.

Output: docs/system-boundary-diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Color palette (kept consistent with the C3 lineage diagram for visual
# coherence across governance artifacts)
# ---------------------------------------------------------------------------

COLORS = {
    "input":       ("#dbeafe", "#1e40af"),    # blue
    "service":     ("#dcfce7", "#166534"),    # green
    "model":       ("#fef3c7", "#92400e"),    # amber
    "monitoring":  ("#e9d5ff", "#6b21a8"),    # purple
    "governance":  ("#fee2e2", "#991b1b"),    # red
    "out_of_scope":("#f3f4f6", "#9ca3af"),    # gray (greyed out — explicitly excluded)
}


def draw_box(ax, x, y, w, h, kind, lines, fontsize=9, fontweight="normal", italic=False, dashed=False):
    fill, edge = COLORS[kind]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.4,
        facecolor=fill,
        edgecolor=edge,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    style = "italic" if italic else "normal"
    ax.text(
        x + w / 2, y + h / 2,
        "\n".join(lines),
        ha="center", va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        fontstyle=style,
        color="#111827",
    )


def draw_arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0.0, 0.10), color="#374151"):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=14),
    )
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label, ha="center", va="center",
            fontsize=7.5, color="#4b5563", style="italic",
        )


def draw_zone(ax, x, y, w, h, label, color="#9ca3af"):
    """Dashed perimeter for the trust zone."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1.6, linestyle=(0, (4, 3)),
        facecolor="none", edgecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + 0.18, y + h - 0.25, label,
            fontsize=10, fontweight="bold", color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---- Title ----
    ax.text(8, 9.55,
            "System Boundary Diagram — Iris Classifier Service",
            ha="center", fontsize=15, fontweight="bold", color="#111827")
    ax.text(8, 9.15,
            "Component 5: scope of risk assessment for the deployed system. Out-of-scope categories are marked.",
            ha="center", fontsize=10, color="#6b7280", style="italic")

    # ---- Trust zone outline (everything inside the dashed border is "the system") ----
    draw_zone(ax, 2.4, 1.6, 11.2, 6.8, "Trust boundary — Iris classifier service")

    # ---- INPUT (left, outside trust zone) ----
    draw_box(ax, 0.2, 6.0, 2.0, 1.4, "input",
             ["Upstream client", "(any HTTP caller)"], fontweight="bold")
    draw_box(ax, 0.2, 4.0, 2.0, 1.4, "input",
             ["Operator", "(metrics scrape,", "config changes)"])
    draw_box(ax, 0.2, 2.0, 2.0, 1.4, "input",
             ["Auditor", "(reads audit", "trail + memos)"])

    # ---- SERVICE LAYER (FastAPI service interior) ----
    draw_box(ax, 2.7, 6.7, 2.4, 1.2, "service",
             ["FastAPI ingress", "Pydantic schema", "validation"], fontweight="bold")
    draw_box(ax, 2.7, 5.0, 2.4, 1.2, "service",
             ["Integrity check", "NaN/Inf reject (400)", "|z|>5 anomaly log"])
    draw_box(ax, 2.7, 3.3, 2.4, 1.2, "service",
             ["Preprocessing", "+ interactions", "StandardScaler"])

    # ---- MODEL LAYER ----
    draw_box(ax, 5.6, 5.3, 2.5, 1.5, "model",
             ["RandomForest (active)", "baseline_run", "M3 run 0ab6a6fa..."], fontweight="bold")
    draw_box(ax, 5.6, 3.3, 2.5, 1.2, "model",
             ["Variant A (rollback)", "high_min_split", "models/variants/"])

    # ---- MONITORING LAYER ----
    draw_box(ax, 8.6, 6.7, 2.5, 1.2, "monitoring",
             ["Prometheus", "(5s scrape, 2d", "retention)"], fontweight="bold")
    draw_box(ax, 8.6, 5.0, 2.5, 1.2, "monitoring",
             ["Grafana dashboard", "PSI / latency /", "anomaly panels"])
    draw_box(ax, 8.6, 3.3, 2.5, 1.2, "monitoring",
             ["Offline drift (C4)", "PSI + KS + impact", "on snapshot windows"])

    # ---- OUTPUT (right, partly inside the boundary) ----
    draw_box(ax, 11.6, 6.0, 1.9, 1.4, "service",
             ["JSON response", "predicted_class", "probabilities"], fontweight="bold")
    draw_box(ax, 11.6, 4.0, 1.9, 1.4, "service",
             ["/metrics endpoint", "(scrape only)"])

    # ---- GOVERNANCE STRIP (bottom) ----
    draw_box(ax, 2.4, 1.85, 11.2, 0.9, "governance",
             ["Governance & Audit (C3): Model card • Risk register • Audit trail (append-only) • Lineage diagram"],
             fontsize=10, fontweight="bold")

    # ---- OUT-OF-SCOPE STRIP (right side) ----
    ax.text(15.4, 8.3, "OUT OF SCOPE",
            ha="center", fontsize=10, fontweight="bold", color="#9ca3af")
    ax.text(15.4, 8.0, "(NOT part of this system)",
            ha="center", fontsize=8, color="#9ca3af", style="italic")
    draw_box(ax, 14.1, 6.7, 2.5, 1.2, "out_of_scope",
             ["LLM / generative", "model"], italic=True, dashed=True)
    draw_box(ax, 14.1, 5.3, 2.5, 1.2, "out_of_scope",
             ["Retriever / vector", "store"], italic=True, dashed=True)
    draw_box(ax, 14.1, 3.9, 2.5, 1.2, "out_of_scope",
             ["Tool execution /", "external APIs"], italic=True, dashed=True)
    draw_box(ax, 14.1, 2.5, 2.5, 1.2, "out_of_scope",
             ["User PII / consent", "store"], italic=True, dashed=True)

    # ---- Arrows ----
    # Client -> ingress
    draw_arrow(ax, 2.2, 6.7, 2.7, 7.2, label="HTTPS POST /predict", label_offset=(0.0, 0.18))
    # Ingress -> integrity check (vertical)
    draw_arrow(ax, 3.9, 6.7, 3.9, 6.2)
    # Integrity -> preprocessing (vertical)
    draw_arrow(ax, 3.9, 5.0, 3.9, 4.5)
    # Preprocessing -> model
    draw_arrow(ax, 5.1, 3.9, 5.6, 5.5, label="scaled features", label_offset=(0.05, 0.12))
    # Model -> response
    draw_arrow(ax, 8.1, 6.0, 11.6, 6.5, label="predicted class +\nprobabilities", label_offset=(0.0, 0.20))
    # Service emits metrics
    draw_arrow(ax, 5.1, 5.6, 8.6, 7.0, label="metric updates", label_offset=(0.0, 0.15))
    # Prometheus -> Grafana
    draw_arrow(ax, 9.85, 6.7, 9.85, 6.2)
    # Grafana / monitoring -> auditor (down-left)
    draw_arrow(ax, 8.6, 3.9, 2.2, 2.4, label="alerts, dashboards", label_offset=(0.0, 0.20))
    # Operator -> Prometheus
    draw_arrow(ax, 2.2, 4.7, 8.6, 7.0, label="scrape", label_offset=(0.0, 0.15))
    # Vertical down-arrows from each major layer to the governance strip
    for x in [3.9, 6.85, 9.85]:
        draw_arrow(ax, x, 3.3, x, 2.75)

    # ---- Legend ----
    legend_handles = [
        mpatches.Patch(color=COLORS["input"][0], label="External actors"),
        mpatches.Patch(color=COLORS["service"][0], label="Service layer (FastAPI)"),
        mpatches.Patch(color=COLORS["model"][0], label="Model artifacts"),
        mpatches.Patch(color=COLORS["monitoring"][0], label="Monitoring (C1, C4)"),
        mpatches.Patch(color=COLORS["governance"][0], label="Governance (C3)"),
        mpatches.Patch(color=COLORS["out_of_scope"][0], label="Out of scope"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              ncol=6, frameon=False, fontsize=9)

    # ---- Footnote ----
    ax.text(8, 0.45,
            "Out-of-scope categories from the rubric (retrieval, hallucination, tool execution) are marked but excluded by design — "
            "this is a tabular classifier with no LLM or agentic component. See docs/governance-review.md for the rationale.",
            ha="center", fontsize=8, color="#6b7280", style="italic", wrap=True)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "system-boundary-diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
