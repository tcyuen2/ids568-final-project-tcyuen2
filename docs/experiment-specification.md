# Experiment Specification — Component 2

A/B test design for promoting a candidate Iris classifier model into the
production serving slot established in Component 1.

## 1. Background and motivation

Two RandomForest configurations from Milestone 3 had **identical offline
accuracy on the M3 test set** but different hyperparameters:

| Variant | M3 run ID | Hyperparameters | M3 offline accuracy |
|---|---|---|---|
| A — `high_min_split` (control, currently serving) | `6e01be84...` | n=100, depth=10, min_split=5 | 0.967 |
| B — `baseline_run` (challenger) | `0ab6a6fa...` | n=50, depth=5, min_split=2 | 0.933 |

The 3.4-point offline accuracy gap was measured on a single 30-sample
stratified test split. Test sets that small have wide confidence intervals,
so the gap may not replicate online. Variant B is also smaller (50 trees
of depth 5 vs. 100 trees of depth 10), which makes it cheaper to serve —
~50% fewer trees to traverse per prediction. **If the online accuracy gap
is small or absent, the operational savings make B a credible
deployment candidate**, not a reject.

This experiment exists to settle that question with a sample size large
enough to make the answer real.

## 2. Hypothesis

Framed as a **non-inferiority test from the operator's perspective** — we
already know A is the safer choice; the question is whether B is good
enough to justify the operational savings.

- **Null (H₀):** the live accuracy of variant B is at least 3 percentage
  points worse than variant A. Formally: `acc_A − acc_B ≥ 0.03`.
- **Alternative (H₁):** the live accuracy of variant B is within 3
  percentage points of variant A. Formally: `acc_A − acc_B < 0.03`.
- **Decision rule:** reject H₀ (i.e., promote B as a deployment option) if
  the upper bound of a 95% one-sided confidence interval on `acc_A − acc_B`
  falls below 0.03. Otherwise either keep A or extend the test.

The 3-percentage-point non-inferiority margin is chosen because:
1. It corresponds to roughly one misclassified prediction in 33 — a
   tolerable level for a non-safety-critical system.
2. It is smaller than the 3.4-point offline gap, so the test will have
   the power to determine whether that gap is real or measurement noise.

## 3. Success and guardrail metrics

**Primary success metric** (decides ship/no-ship):

- **Accuracy on labeled live traffic.** Computed per-variant as
  `correct_predictions / total_predictions`. Live labels are available
  in this synthetic system because traffic is generated from a known
  per-class distribution; in a real deployment, ground truth would
  arrive via a delayed feedback channel (e.g., user corrections or a
  human review queue).

**Guardrail metrics** (any one fires → halt the experiment, do not ship B):

| Guardrail | Threshold | Source |
|---|---|---|
| Variant B p99 latency | > 50 ms (5× variant A p99) | Component 1 latency histogram |
| Variant B error rate | > 1% of B's traffic | Component 1 `iris_requests_total` 5xx |
| Variant B mean prediction confidence | drops by >5 percentage points vs. A | Component 1 `iris_prediction_confidence` |
| Variant B feature-drift PSI | crosses 0.25 on any feature *only on B's traffic* | Component 1 `iris_feature_drift_psi` |

The confidence-drop guardrail is included because confidence drops
typically *precede* accuracy drops by hours or days in real systems —
catching it early lets us halt before user-visible damage.

These guardrails are evaluated alongside the primary metric by reading
from the Component 1 Prometheus instance, so the same telemetry that
serves operational monitoring also serves experiment safety.

## 4. Randomization method

**Per-request hash-based bucketing.** Each incoming request is assigned
to A or B by:

```
bucket = "B" if (hash(request_id) % 100) < 50 else "A"
```

where `request_id` is a stable per-request UUID generated at the API
gateway. This gives a 50/50 split with the following guarantees:

- **Deterministic:** the same request always hits the same variant —
  important for retries and idempotency.
- **Independent:** consecutive requests are assigned independently,
  so user-level correlations don't bias either arm.
- **No drift over time:** the split ratio doesn't shift unless we
  deliberately change the threshold.

For the simulation, the same logic is applied per-sample using a
seeded numpy RNG — see `src/ab_test/simulation.py`.

A 50/50 split is chosen rather than the more common 90/10 because the
challenger here is an *older* model with known offline performance, not
an unproven new model. We are willing to expose half of traffic to it
because the downside risk is bounded.

## 5. Sample size and duration

**Power calculation (two-proportion z-test, α = 0.05 two-sided, power = 0.80):**

| Minimum detectable effect (MDE) | Samples per arm | Total | Time at 5 req/s |
|---|---|---|---|
| 0.5% absolute drop | 21,474 | 42,948 | 2.4 hours |
| 1.0% absolute drop | 5,715 | 11,430 | 38.1 minutes |
| 2.0% absolute drop | 1,594 | 3,188 | 10.6 minutes |
| 3.0% absolute drop | 778 | 1,556 | 5.2 minutes |

**Chosen MDE: 1.0%.** Justification:

- Smaller than the 3% non-inferiority margin (so the test can reliably
  distinguish "within margin" from "at margin").
- Larger than what would require an impractically long run (0.5% MDE
  needs 2.4 hours of synthetic traffic at 5 req/s, which produces
  diminishing returns for a teaching system).
- Large enough that real production traffic at typical rates would
  reach this sample size in a few hours rather than days.

**Effect-size derivation (Cohen's h, since we're comparing two proportions):**

```
p1 = 0.967  (control accuracy from M3)
p2 = p1 - 0.010 = 0.957  (control − MDE)
h  = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2))) ≈ 0.0524
n_per_arm = ((z_{α/2} + z_β) / h)^2 ≈ 5,715
```

Computed via `statsmodels.stats.power.NormalIndPower.solve_power`.

**Required sample size:** **5,715 per arm = 11,430 total.**

**Duration:** at 5 req/s (the rate used in the simulation), **38 minutes
of continuous traffic**. With 50/50 splitting this means each variant
sees ~2.5 req/s, so each variant accumulates 5,715 samples over the same
38 minutes.

**Stopping rules:** the experiment stops at *whichever comes first*:

1. The primary-metric sample size (5,715/arm) is reached.
2. Any guardrail metric crosses its threshold for two consecutive 5-minute
   windows.

Sequential testing is **not** used (no peeking adjustments) because the
simulation runs to a fixed sample size in batch. In a real online
deployment, sequential testing with appropriate alpha-spending would be
required.

## 6. Multiple comparisons

This experiment tests **one primary metric** (accuracy) plus a small
number of fixed guardrails. We are not running multiple primary metrics
simultaneously, so no Bonferroni or Benjamini-Hochberg correction is
applied. If we extended the experiment to test multiple primary metrics
in parallel (e.g., accuracy *and* macro-F1), we would adjust alpha to
0.025 to maintain family-wise error rate at 0.05.

## 7. Pre-registered analysis plan

To prevent post-hoc analysis fishing, the analysis is pre-specified
*before* simulation results are observed:

1. **Data:** all requests assigned to A or B, with predicted label and
   ground-truth label, restricted to requests within the first 5,715
   per-arm assignments.
2. **Test:** two-proportion z-test on accuracy (correct/total) for A
   vs. B, two-sided.
3. **Confidence interval:** 95% Wald CI on the difference `acc_A − acc_B`.
4. **Decision:**
   - Ship B if the upper bound of the CI on `acc_A − acc_B` is
     below 0.03 (i.e., we have evidence B is non-inferior to A by the
     3% margin).
   - Otherwise, do not ship; either keep A or run more data depending
     on the CI width.
5. **Guardrails:** report each guardrail metric separately; any single
   guardrail breach overrides the primary-metric decision.

Implementation: `src/ab_test/analyze.py`.

## 8. Reproducibility

- Numpy RNG seeded at 42 throughout the simulation.
- Both model artifacts bundled in `models/variants/` (lineage to M3
  runs `6e01be84...` and `0ab6a6fa...` documented in the model card).
- Single command to reproduce: `python -m src.ab_test.simulation && python -m src.ab_test.analyze`.

---

**Builds on Milestone 3:** the offline experiment runs in M3's
`experiment_results.json` informally compared 5 hyperparameter configs.
This component formalizes that comparison into a statistically rigorous
A/B test for two of those configs. The same M3 quality gates from
`model_validation.py` (accuracy ≥ 0.90, F1 ≥ 0.85, AUC ≥ 0.90) appear
here as guardrail-style sanity checks before any candidate could be
declared a winner.
