"""
Loader for training-distribution reference statistics.

The serving service queries this object to:
  - Look up bin edges and reference proportions for live PSI computation
  - Get expected mean / std / range for input integrity checks
  - Get training class proportions for output drift comparison

The underlying JSON file is produced by `generate_reference.py` once,
checked into the repo, and never modified at runtime. This keeps drift
detection deterministic and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureReference:
    """Per-feature training distribution summary."""

    name: str
    mean: float
    std: float
    min: float
    max: float
    bin_edges: list[float]
    bin_pcts: list[float]


class Reference:
    """Container for full reference statistics. Immutable after load."""

    def __init__(self, path: Path | str):
        with open(path) as f:
            data = json.load(f)

        self.feature_names: list[str] = data["feature_names"]
        self.psi_n_bins: int = data["psi_n_bins"]
        self.class_pcts: dict[int, float] = {
            int(k): float(v) for k, v in data["class_pcts"].items()
        }
        self.class_names: list[str] = data["class_names"]

        self._features: dict[str, FeatureReference] = {}
        for name, stats in data["features"].items():
            self._features[name] = FeatureReference(
                name=name,
                mean=stats["mean"],
                std=stats["std"],
                min=stats["min"],
                max=stats["max"],
                bin_edges=stats["bin_edges"],
                bin_pcts=stats["bin_pcts"],
            )

    def feature(self, name: str) -> FeatureReference:
        return self._features[name]

    def class_pcts_array(self) -> list[float]:
        """Class proportions in label order [0, 1, 2]."""
        return [self.class_pcts[i] for i in sorted(self.class_pcts.keys())]
