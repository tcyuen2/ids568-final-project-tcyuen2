"""
Statistical analysis of A/B simulation results.

Implements the pre-registered analysis plan from
docs/experiment-specification.md, section 7:

  1. Load simulation results (per-request log)
  2. Two-proportion z-test on accuracy (A vs B), two-sided
  3. 95% Wald CI on the difference (acc_A − acc_B)
  4. Apply the decision rule:
       - Ship B if upper CI bound on (acc_A − acc_B) < 0.03
       - Otherwise: keep A or run more data
  5. Print results and write a structured summary to JSON

Usage:
    python -m src.ab_test.analyze
    python -m src.ab_test.analyze --results path/to/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "src" / "ab_test" / "results.json"
DEFAULT_OUTPUT = REPO_ROOT / "src" / "ab_test" / "analysis.json"

# Pre-registered decision parameters from the spec.
ALPHA = 0.05
NON_INFERIORITY_MARGIN = 0.03  # 3 percentage points


@dataclass
class TwoProportionResult:
    """Output of the two-proportion z-test + CI."""

    n_a: int
    correct_a: int
    p_a: float
    n_b: int
    correct_b: int
    p_b: float
    diff: float                # p_a - p_b
    se_diff: float             # standard error of the difference
    z: float                   # test statistic
    p_value_two_sided: float
    ci_low_95: float           # lower bound of 95% Wald CI on diff
    ci_high_95: float          # upper bound


def two_proportion_test(correct_a: int, n_a: int, correct_b: int, n_b: int) -> TwoProportionResult:
    """Two-proportion z-test with 95% Wald CI on the difference.

    Uses the pooled-variance form for the test statistic and the
    separate-variance form for the CI. This is the standard textbook
    approach (Newcombe-Wilson would be more conservative for small
    samples; for n in the thousands, Wald is fine).
    """
    if n_a == 0 or n_b == 0:
        raise ValueError("Both arms must have observations.")

    p_a = correct_a / n_a
    p_b = correct_b / n_b
    diff = p_a - p_b

    # Pooled SE for the test statistic (assumes H0: p_a == p_b)
    p_pool = (correct_a + correct_b) / (n_a + n_b)
    se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Separate-variance SE for the CI (better when proportions differ)
    se_diff = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_critical = stats.norm.ppf(1 - ALPHA / 2)  # 1.96 for alpha=0.05
    ci_low = diff - z_critical * se_diff
    ci_high = diff + z_critical * se_diff

    return TwoProportionResult(
        n_a=n_a, correct_a=correct_a, p_a=p_a,
        n_b=n_b, correct_b=correct_b, p_b=p_b,
        diff=diff, se_diff=se_diff,
        z=z, p_value_two_sided=p_value,
        ci_low_95=ci_low, ci_high_95=ci_high,
    )


def make_decision(test: TwoProportionResult) -> dict:
    """Apply the pre-registered decision rule."""
    # Decision: ship B if we have evidence acc_A − acc_B < 0.03.
    # Equivalent to: upper bound of CI on (acc_A − acc_B) is below the margin.
    upper_bound = test.ci_high_95
    margin = NON_INFERIORITY_MARGIN

    if upper_bound < margin:
        decision = "SHIP_B"
        rationale = (
            f"Upper bound of 95% CI on (acc_A − acc_B) is {upper_bound:+.4f}, "
            f"which is below the non-inferiority margin of {margin:+.4f}. "
            f"This is sufficient evidence that variant B is non-inferior to "
            f"variant A by the pre-registered margin."
        )
    else:
        # Decision: either keep A or run more data, depending on whether
        # the CI shows B might still be acceptable but we lack power.
        # If the CI is wide (high - low > 0.02), more data could help.
        # If the CI excludes the margin (lower bound > margin), keep A.
        ci_width = test.ci_high_95 - test.ci_low_95
        if test.ci_low_95 > margin:
            decision = "KEEP_A"
            rationale = (
                f"Lower bound of 95% CI on (acc_A − acc_B) is {test.ci_low_95:+.4f}, "
                f"above the non-inferiority margin of {margin:+.4f}. We have "
                f"evidence variant B is materially worse than A on this metric."
            )
        elif ci_width > 0.02:
            decision = "RUN_MORE_DATA"
            rationale = (
                f"95% CI on (acc_A − acc_B) is [{test.ci_low_95:+.4f}, {test.ci_high_95:+.4f}], "
                f"a width of {ci_width:.4f}. The CI straddles the margin, so the "
                f"current sample is insufficient to decide. Extend the experiment."
            )
        else:
            decision = "KEEP_A"
            rationale = (
                f"95% CI on (acc_A − acc_B) is [{test.ci_low_95:+.4f}, {test.ci_high_95:+.4f}]. "
                f"The CI is tight but its upper bound {test.ci_high_95:+.4f} exceeds "
                f"the margin {margin:+.4f}. Insufficient evidence to ship B."
            )

    return {
        "decision": decision,
        "rationale": rationale,
        "test": asdict(test),
        "alpha": ALPHA,
        "non_inferiority_margin": margin,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    s_a = results["summary"]["A"]
    s_b = results["summary"]["B"]
    logger.info("Loaded results from %s", args.results)
    logger.info("Variant A (high_min_split):   %d/%d correct (%.4f)",
                s_a["correct"], s_a["total"], s_a["accuracy"])
    logger.info("Variant B (baseline_run):     %d/%d correct (%.4f)",
                s_b["correct"], s_b["total"], s_b["accuracy"])

    test = two_proportion_test(s_a["correct"], s_a["total"],
                               s_b["correct"], s_b["total"])

    print("\n" + "=" * 68)
    print(" A/B TEST RESULTS — Iris Classifier (M3-derived variants)")
    print("=" * 68)
    print(f" Variant A (control, high_min_split):  acc = {test.p_a:.4f}  (n={test.n_a:,})")
    print(f" Variant B (challenger, baseline_run): acc = {test.p_b:.4f}  (n={test.n_b:,})")
    print(f" Difference (A − B):                   {test.diff:+.4f}")
    print(f" Standard error of difference:         {test.se_diff:.4f}")
    print(f" Two-proportion z-test:                z = {test.z:+.4f}")
    print(f" p-value (two-sided):                  {test.p_value_two_sided:.4f}")
    print(f" 95% CI on (acc_A − acc_B):            [{test.ci_low_95:+.4f}, {test.ci_high_95:+.4f}]")

    decision = make_decision(test)
    print(f"\n DECISION: {decision['decision']}")
    print(f" RATIONALE: {decision['rationale']}")
    print("=" * 68 + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(decision, f, indent=2)
    logger.info("Wrote analysis to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
