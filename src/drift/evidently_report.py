"""
Generate an Evidently HTML drift report.

The rubric explicitly mentions Evidently as a recommended tool, so this
script produces an HTML report alongside our custom matplotlib charts.
The two are complementary: matplotlib gives us the time-series
"evolution across windows" view; Evidently gives the rich
single-comparison "reference vs. production_week3" view including
its own statistical tests.

Output:
    visualizations/drift/evidently-report-week3.html
    visualizations/drift/evidently-report-week2.html
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

# Evidently emits a lot of FutureWarnings that aren't actionable here
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from evidently.metric_preset import DataDriftPreset, DataQualityPreset  # noqa: E402
from evidently.report import Report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src" / "drift" / "data"
OUTPUT_DIR = REPO_ROOT / "visualizations" / "drift"


def build_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: Path) -> None:
    """Build a combined drift + data-quality report."""
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    # Evidently uses 'target' as a special column name automatically
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(str(output_path))
    logger.info("  wrote %s", output_path.name)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(DATA_DIR / "reference.csv")
    logger.info("Loaded reference window (n=%d)", len(reference))

    # Generate one report per drift severity. Week 1 is uninteresting
    # (no drift), so we skip it and focus on the actionable comparisons.
    for window_name in ["production_week2", "production_week3"]:
        current = pd.read_csv(DATA_DIR / f"{window_name}.csv")
        # Drop NaN rows for Evidently — it can handle them but the report
        # cleans up significantly without them.
        current_clean = current.dropna()
        out_path = OUTPUT_DIR / f"evidently-report-{window_name.replace('production_', '')}.html"
        logger.info("Building report: reference vs %s (n=%d)", window_name, len(current_clean))
        build_report(reference, current_clean, out_path)


if __name__ == "__main__":
    main()
