# Experiment Specification — Component 2

A/B test design for promoting a candidate Iris classifier model into the
production serving slot from Component 1.

## 1. Background

Two RandomForest configs from Milestone 3 had **identical offline
accuracy on the M3 test set** but different hyperparameters:

| Variant | M3 run ID | Hyperparameters | M3 offline accuracy |
|---|---|---|---|
| A — `high_min_split` (control, currently serving) | `6e01be84...` | n=100, depth=10, min_split=5 | 0.967 |
| B — `baseline_run` (challenger) | `0ab6a6fa...` | n=50, depth=5, min_split=2 | 0.933 |

The 3.4-point gap was measured on a 30-sample test set, which has wide
CIs — that gap might not replicate online. B is also smaller (50 trees
vs. 100), so it's cheaper to serve. **If the live accuracy gap is small
or absent, B becomes a credible deployment candidate.** This experiment
exists to settle that question with a sample size that makes the answer
real.

## 2. Hypothesis


- **Null:** B's live accuracy is at least 3 percentage points wors
- **Alternative:** B is within 3 percentage points of A.
- **Decision:** reject H₀ (i.e., promote B) if the upper bound of a 95% one-sided CI on `acc_A − acc_B` is below 0.03. Otherwise keep A or extend the test.

The 3-point margin works out to roughly 1 misclassified prediction in
33 — tolerable here, smaller than the 3.4-point offline gap so the test
has the power to determine whether the gap is real or noise.

## 3. Metrics

**Primary metric**

- **Accuracy on labeled live traffic.** `correct / total` per variant.
  Live labels are available because traffic comes from a known
  per-class distribution; in real production, labels would arrive
  through delayed feedback (user corrections or human review).

**Guardrail metrics** (any one fires → halt the experiment, do not ship B):

| Guardrail | Threshold | Source |
|---|---|---|
| B p99 latency | > 50 ms (5× A's) | C1 latency histogram |
| B error rate | > 1% of B's traffic | C1 `iris_requests_total` 5xx |
| B mean prediction confidence | drops > 5 pp vs. A | C1 `iris_prediction_confidence` |
| B feature-drift PSI | crosses 0.25 on any feature on B's traffic only | C1 `iris_feature_drift_psi` |

The confidence-drop guardrail is here because confidence drops usually
*precede* accuracy drops by hours or days in real systems — catching it
early lets us halt before user-visible damage.

## 4. Randomization

**Per-request hash-based bucketing.** Each request gets bucketed by:

```
bucket = "B" if (md5(request_id) % 100) < 50 else "A"
```

Where `request_id` is a UUID generated at the API gateway. This gives
50/50 split with three properties:

- **Deterministic:** same request → same variant (matters for retries)
- **Independent:** consecutive requests assigned independently
- **Stable over time:** ratio doesn't shift unless we change the threshold

50/50 is chosen rather than the more common 90/10 because B is an
already-trained model with known offline performance, not an unproven
new model. The downside risk is bounded.

For the simulation, the same logic is applied per-sample with a seeded
numpy RNG — see `src/ab_test/simulation.py`.

## 5. Sample size and duration

**Power calculation** (two-proportion z-test, α=0.05 two-sided, power=0.80):

| MDE (absolute drop) | n per arm | Total | Time at 5 req/s |
|---|---|---|---|
| 0.5% | 21,474 | 42,948 | 2.4 hours |
| **1.0% (chosen)** | **5,715** | **11,430** | **38.1 minutes** |
| 2.0% | 1,594 | 3,188 | 10.6 minutes |
| 3.0% | 778 | 1,556 | 5.2 minutes |

**Why 1.0%:** smaller than the 3% non-inferiority margin (so the test
can reliably distinguish "within margin" from "at margin"), but large
enough to be practical to run.

**Effect-size derivation (Cohen's h):**

```
p1 = 0.967  (control accuracy from M3)
p2 = 0.957  (control − MDE)
h  = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2))) ≈ 0.0524
n_per_arm ≈ 5,715
```

Computed via `statsmodels.stats.power.NormalIndPower.solve_power`.

**Required sample:** **5,715 per arm = 11,430 total.**
**Duration:** 38 minutes at 5 req/s.

**Stopping rules:**

1. Stop when the per-arm sample size (5,715) is reached, **or**
2. Stop early if any guardrail crosses its threshold for two consecutive
   5-min windows.

Sequential testing isn't used (no peeking adjustments) because the
simulation runs to a fixed sample size in batch. A real online
deployment would need sequential testing with alpha-spending.

## 6. Multiple comparisons

One primary metric (accuracy) plus a small number of fixed guardrails.


## 7. Pre-registered analysis plan

Pre-specified *before* simulation runs to prevent post-hoc fishing:

1. **Data:** all requests assigned to A or B with predicted and
   ground-truth labels, restricted to the first 5,715 per-arm.
2. **Test:** two-proportion z-test on accuracy, two-sided.
3. **CI:** 95% Wald CI on `acc_A − acc_B`.
4. **Decision:**
   - Ship B if upper bound of CI on `acc_A − acc_B` < 0.03
   - Otherwise: keep A or run more data depending on CI width
5. **Guardrails:** report each separately; any single breach overrides
   the primary-metric decision.

Implementation: `src/ab_test/analyze.py`.

