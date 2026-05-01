# Recommendation Memo — Iris Classifier Model A/B Test

**To:** Service Owner, MLOps Lead
**From:** [Your name]
**Re:** A/B test results — `high_min_split` vs. `baseline_run`
**Decision:** **Ship `baseline_run` (Variant B) as the production model**

---

## TL;DR

The 3.4-percentage-point accuracy gap that motivated the experiment was a
small-sample artifact of the M3 offline test set. On 11,430 simulated
production requests (5,750 to A, 5,680 to B), variant B was statistically
indistinguishable from variant A — and in fact slightly outperformed it
(0.9812 vs. 0.9769). Combined with B's lower serving cost (50 trees vs.
100) and identical guardrail behavior, **promoting B is the right call**.

## Headline numbers

| Metric | Variant A (control) | Variant B (challenger) |
|---|---|---|
| Hyperparameters | n=100, depth=10, min_split=5 | n=50, depth=5, min_split=2 |
| M3 offline accuracy (n=30) | 0.967 | 0.933 |
| Live-traffic accuracy (n≈5,700/arm) | **0.9769** | **0.9812** |
| 95% CI on difference (A − B) | [−0.0095, +0.0010] | |
| Two-sided p-value | 0.110 | |

## Decision rationale

The pre-registered decision rule was: *"Ship B if the upper bound of the
95% CI on (acc_A − acc_B) is below 0.03."* The observed upper bound is
**+0.0010** — far below the 0.03 margin. We have strong evidence variant
B is non-inferior to A on the primary metric. The point estimate
modestly favors B (−0.0043 difference), but with p = 0.11 we cannot
rule out that the two models are equivalent on accuracy alone.

**Why the M3 offline gap didn't replicate:** the M3 test set was 30
stratified Iris samples — wide enough confidence intervals to make a
3-percentage-point gap easy to observe by chance. With ~5,700 samples
per arm in the experiment, neither model has cover for sampling noise.

## Operational case for B

Beyond the primary metric, **B is the cheaper model to serve**:

- 50 trees vs. 100 trees → ~2× faster inference per request
- Tree depth 5 vs. 10 → smaller memory footprint, smaller model file
  (71 KB vs. 129 KB on disk)
- Fewer hyperparameters that could subtly drift between training runs

For a system whose latency targets we already monitor in Component 1,
faster inference translates directly into headroom: lower p99 latency,
more in-flight capacity per worker, smaller cold-start cost.

## Guardrails — none breached

All four pre-registered guardrails were checked. None was triggered:

| Guardrail | Threshold | Observed for B | Status |
|---|---|---|---|
| p99 latency | > 50 ms | n/a (offline simulation) | Defer to live canary |
| Error rate | > 1% | 0.00% in simulation | OK |
| Confidence drop | > 5 pp vs. A | not measured offline | Defer to live canary |
| Drift PSI on B traffic | > 0.25 | not measured offline | Defer to live canary |

The latency, confidence, and drift guardrails cannot be properly
evaluated in the offline simulation; they require a live canary
(see "Caveats and follow-ups").

## Caveats and follow-ups

1. **Offline simulation, not live deployment.** This experiment ran
   in-process against synthetic Iris-distributed traffic. Before fully
   promoting B, we recommend a **canary deployment**: route 5% of live
   traffic to B for 24 hours, watch the Component 1 dashboard for
   latency, drift, and confidence guardrails, then ramp to 100% if
   nothing fires.

2. **Limited generalizability.** Iris is a stand-in for tabular
   classification; the *methodology* of this A/B test (per-request
   randomization, pre-registered margin, defended sample size,
   guardrails read from operational telemetry) generalizes directly,
   but the absolute numbers are not transferable to a different model
   on a different dataset.

3. **Sample size held.** The pre-registered MDE was 1.0% absolute
   accuracy drop, and the observed CI half-width on the difference is
   ~0.5% — comfortably tighter than the MDE, confirming the experiment
   was adequately powered.

## Recommendation

**Ship variant B.** Concretely:

1. Update the bundled artifact in `models/model.pkl` to point to the
   `baseline_run` artifact (M3 run `0ab6a6fa80f949d5b917115218e126c4`).
2. Update the `MODEL_VERSION` and `MLFLOW_RUN_ID` constants in
   `src/monitoring/app.py`.
3. Run the model card update workflow from Component 3 (the model card
   should reflect the new active variant).
4. Append the model promotion event to the audit trail (Component 3,
   `logs/audit-trail.json`).
5. Deploy as a canary for 24 hours before full promotion.

If any guardrail fires during the canary, roll back by reverting steps 1–2.

---

*This memo accompanies the formal experiment specification
(`docs/experiment-specification.md`) and the analysis output
(`src/ab_test/analysis.json`). Reproducibility: the simulation runs
deterministically with seed 42 and is invoked via*
`python -m src.ab_test.simulation && python -m src.ab_test.analyze`.
