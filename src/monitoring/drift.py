"""
Drift computation utilities.

Two pieces:
  1. `psi(expected_pcts, actual_pcts)`: pure function implementing the
     Population Stability Index formula. Used both at serving time
     (live drift in C1) and in offline analysis (C4).
  2. `RollingWindow`: thread-safe sliding window of recent observations
     per feature, used to compute live PSI without keeping unbounded history.

PSI formula:
    PSI = sum over bins of (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

Interpretation thresholds (Siddiqi, "Credit Risk Scorecards"):
    PSI < 0.1  : stable
    0.1 - 0.25 : moderate drift, investigate
    PSI > 0.25 : significant drift, action required

These same thresholds drive both the live alerts in C1 and the
diagnostic analysis in C4 — keeping them consistent is what lets us
say the dashboard would have caught the offline-analyzed drift.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Iterable

import numpy as np

# Smoothing constant: PSI is undefined when a bin has zero mass on either
# side (log(0) blows up). We replace zeros with this small value, which is
# the standard practice and matches what generate_reference.py uses.
_EPSILON = 1e-6


def psi(expected_pcts: np.ndarray, actual_pcts: np.ndarray) -> float:
    """Compute Population Stability Index between two binned distributions.

    Both inputs must be 1-D arrays of bin proportions over the *same*
    bin edges, summing to ~1.0. Returns a single float.
    """
    expected = np.asarray(expected_pcts, dtype=float)
    actual = np.asarray(actual_pcts, dtype=float)

    if expected.shape != actual.shape:
        raise ValueError(
            f"PSI requires matching shapes; got {expected.shape} vs {actual.shape}"
        )

    # Smooth zeros to avoid log(0). Doesn't materially shift PSI for
    # non-degenerate distributions.
    expected = np.where(expected == 0, _EPSILON, expected)
    actual = np.where(actual == 0, _EPSILON, actual)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def histogram_pcts(values: Iterable[float], bin_edges: list[float]) -> np.ndarray:
    """Bin `values` according to `bin_edges` and return proportions per bin.

    Used to compute the "actual" side of PSI from a live observation window.
    The bin edges come from the reference stats — same edges as training,
    so the resulting proportions are directly comparable.
    """
    values_arr = np.asarray(list(values), dtype=float)
    if values_arr.size == 0:
        # Nothing observed yet — return uniform-ish so PSI ~= 0 until we
        # have data. Avoids spurious "drift!" alerts on a cold start.
        n_bins = len(bin_edges) - 1
        return np.full(n_bins, 1.0 / n_bins)

    counts, _ = np.histogram(values_arr, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        n_bins = len(bin_edges) - 1
        return np.full(n_bins, 1.0 / n_bins)
    return counts / total


class RollingWindow:
    """Thread-safe fixed-capacity buffer of recent values per feature.

    The serving service appends each incoming feature value, and on a
    periodic schedule pulls the buffer contents to recompute PSI / mean / std.

    Why a deque with maxlen instead of a true streaming algorithm?
      - Iris is a low-throughput teaching example; memory is not a concern.
      - A literal window of recent observations is the easiest thing to
        reason about and explain in the interpretation document.
      - For a real high-throughput service, a streaming PSI variant (e.g.,
        with reservoir sampling) would replace this — see drift-diagnostic-
        report.md for the discussion of this trade-off.
    """

    def __init__(self, feature_names: list[str], capacity: int = 500):
        self.capacity = capacity
        self.feature_names = feature_names
        self._buffers: dict[str, deque[float]] = {
            name: deque(maxlen=capacity) for name in feature_names
        }
        self._predictions: deque[int] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, feature_values: dict[str, float], predicted_class: int) -> None:
        """Append one observation: per-feature scaled values + predicted class."""
        with self._lock:
            for name, val in feature_values.items():
                if name in self._buffers:
                    self._buffers[name].append(val)
            self._predictions.append(predicted_class)

    def snapshot_features(self) -> dict[str, list[float]]:
        """Return a copy of current buffer contents per feature.

        Copying under the lock so the caller can compute PSI / mean / std
        without blocking the request path.
        """
        with self._lock:
            return {name: list(buf) for name, buf in self._buffers.items()}

    def snapshot_predictions(self) -> list[int]:
        with self._lock:
            return list(self._predictions)

    def size(self) -> int:
        with self._lock:
            return len(self._predictions)
