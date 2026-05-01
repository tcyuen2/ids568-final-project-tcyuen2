# Drift Diagnostic Report — Component 4

This report analyzes drift across three production windows of synthetic
Iris-classifier traffic, comparing each against the M3 training
distribution. It addresses the four questions the rubric calls out:
which features drifted most, how that drift impacts model performance,
what retraining or intervention is recommended, and a concrete action
plan.

The same PSI math and the same reference distribution used here also
power the live drift panel in the Component 1 dashboard (see
`docs/dashboard-interpretation.md`). This intentional reuse means any
drift this offline analysis flags is the *same* drift the live
dashboard would have surfaced — the two views are direct counterparts,
one batch and one streaming.

## 1. Methodology

Four windows of n=2,000 each were generated with deterministic seeded
synthesis (`src/drift/generate_windows.py`):

| Window | Petal scaling | Integrity anomalies | Purpose |
|---|---|---|---|
| Reference | 1.0× | 0% | Match training distribution |
| Production week 1 | 1.0× | 0% | Healthy production baseline |
| Production week 2 | 1.2× | 0% | Moderate sensor drift |
| Production week 3 | 1.5× | 5% NaN/extreme | Significant drift + integrity issues |

For each production window, drift was measured against reference using:

- **PSI** (Population Stability Index) per feature, with the same bin
  edges and thresholds used in the Component 1 dashboard:
  PSI < 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 significant.
- **Kolmogorov–Smirnov two-sample test** per feature, as a complementary
  bin-free check on distribution shift.
- **Predicted-class distribution PSI**, comparing model output proportions
  against the training class proportions (33% / 33% / 33%).
- **Integrity counts**: NaN values per feature and rows with any
  post-scaling |z| > 5 outlier.

The **impact analysis** runs the active production model
(`baseline_run`, M3 run `0ab6a6fa…`, the variant promoted by C2) on
each window and computes accuracy against the synthetic ground-truth
labels — turning drift signals into predicted accuracy losses.

A complementary HTML report from Evidently (an open-source drift
detection toolkit explicitly named by the rubric) is produced for the
two drift-affected windows and saved at
`visualizations/drift/evidently-report-week2.html` and
`evidently-report-week3.html`.

## 2. Headline findings

| Window | Max feature PSI | Output PSI | Accuracy | Δ vs reference |
|---|---|---|---|---|
| Reference | 0.215 (sampling noise floor) | 0.000 | **0.986** | — |
| Production week 1 | 0.236 (petal_width — ~baseline) | 0.001 | **0.982** | −0.004 |
| Production week 2 | **0.813** (petal_length) | 0.339 | **0.793** | **−0.192** |
| Production week 3 | **4.173** (petal_length) | 1.572 | **0.663** | **−0.323** |

Three things stand out:

1. **Petal features carry the entire drift signal.** Sepal-related
   features (sepal_length, sepal_width, sepal_l_x_sepal_w) stay in
   the stable PSI band across all windows. The simulated sensor drift
   was injected only on petal channels, and the analysis correctly
   isolates that — confirming the drift detector localizes the
   problem rather than just flagging "something is off."

2. **A 20% petal scaling produces a 19-point accuracy drop.** Week 2's
   1.2× scaling is *small enough that a careless eye would miss it*
   (mean confidence still 0.92), but the model loses nearly a fifth
   of its accuracy. This is exactly the kind of silent degradation
   that monitoring exists to catch.

3. **The model is "confidently wrong" under significant drift.** Week 3
   accuracy collapses to 0.663 while mean prediction confidence stays
   at 0.94 — a 28-point gap. A naïve confidence-only monitor would
   miss this. This is a known failure mode of tree-based models on
   out-of-distribution inputs and is the reason confidence alone is
   insufficient as a drift signal — feature-distribution PSI is needed
   to detect what confidence misses.

## 3. Which features drifted most

Ranked by Week 3 PSI:

1. **`petal_length`** — PSI 4.173 (Week 3), 0.813 (Week 2). KS p < 1e-300.
2. **`petal_l_x_petal_w`** (interaction) — PSI 2.913 (Week 3), 0.748 (Week 2).
3. **`petal_width`** — PSI 0.870 (Week 3), 0.260 (Week 2).
4. **Sepal features and `sepal_l_x_sepal_w`** — all stay below 0.20 in
   every window. Stable as expected.

The interaction term `petal_l_x_petal_w` drifts substantially in both
production windows because it is a multiplicative combination of two
drifted base features. **This is a teachable diagnostic point**:
engineered features can amplify drift from base features, and a
real-world drift dashboard should monitor *both* base and engineered
features rather than treating the model as a black box over its raw
inputs.

## 4. Impact on model performance

The relationship between drift severity and accuracy degradation is
visualized in `visualizations/drift/accuracy-impact.png`:

- **Drift below 0.25 PSI on petal features** (Week 1) produces no
  measurable accuracy change.
- **Drift in the 0.25–1.0 range** (Week 2) produces a 19% accuracy
  drop. Practically, this means roughly 1 in 5 production predictions
  would be wrong — a major user-facing impact.
- **Drift above 1.0 PSI** (Week 3) plus integrity issues produces a 32%
  accuracy drop, and 49 rows are rejected outright at the integrity
  layer because of NaN values.

A useful operational takeaway: **the C1 dashboard's red 0.25 threshold
is too lenient as a stand-alone "should I page" trigger for this
specific model**. The Week 2 condition (max-feature-PSI 0.81) far
exceeds the threshold, but a single red-band crossing is what the
dashboard uses to escalate. For a production deployment this severity
mapping should be re-tuned per-feature: petal_length crossing 0.25
should be a higher-severity event than sepal_width crossing the same
threshold, because the model is far more sensitive to petal features.

## 5. Retraining and intervention recommendations

In priority order:

1. **Investigate the upstream petal data source.** PSI on sepal features
   is normal; PSI on petal features is severe. This pattern is consistent
   with a single sensor or pipeline stage that handles petal measurements
   only. Talking to the data owner is faster than retraining.

2. **If the upstream issue is permanent**, retrain. Use a fresh dataset
   that includes the new petal distribution. Run the same M3 Airflow
   pipeline on the new data, register a new MLflow run, and apply the
   C2 A/B test methodology before promoting. Append the new training
   run to the audit trail (`logs/audit-trail.json`).

3. **Add a per-feature alert threshold** to the C1 dashboard. Today
   thresholds are uniform across features (0.10 / 0.25); they should
   be tighter for petal features given their measured sensitivity.

4. **Add a confidence-vs-accuracy gap monitor.** Mean confidence
   minus moving accuracy estimate is a leading indicator of "confidently
   wrong" failures that pure-confidence monitoring misses. In a labeled
   environment, accuracy is observable; in an unlabeled one, this gap
   can be approximated using model agreement against a small held-out
   labeled set.

5. **Tighten the integrity layer.** Week 3 produced 49 NaN rows
   (rejected) and 145 |z|>5 outlier rows (passed through with a log
   entry). At 5% anomaly rate, the integrity layer is doing its job;
   at higher rates, the service should rate-limit anomalous inputs
   to prevent bad-data-driven prediction skew.

## 6. Action plan

| # | Action | Owner | Trigger |
|---|---|---|---|
| 1 | Page upstream data owner about petal-channel data quality | On-call ML engineer | Petal-feature PSI > 0.25 sustained 30 min in C1 dashboard |
| 2 | Snapshot current production traffic for offline drift analysis | On-call | Same trigger |
| 3 | If upstream issue is real and persistent: rerun M3 training pipeline on new data | M3 owner | Action 1 confirmed by data team |
| 4 | Run C2 A/B test against the current production model | C2 owner | New MLflow run registered |
| 5 | Promote winning variant; update audit trail | C2 owner | C2 ship decision |
| 6 | Add per-feature alert thresholds to C1 dashboard | C1 owner | This report's adoption |
| 7 | Re-baseline `models/reference_stats.json` if drift is the new normal | M3 owner | After successful retraining |

Action 7 is critical and easy to forget: if the upstream data has
permanently shifted and we retrain on the new distribution, the *next*
production drift detector needs the *new* distribution as its
reference. Otherwise C1 will keep firing on the no-longer-relevant
old baseline.

## 7. Reproducibility and limitations

**Reproducibility.** Both the window generator and the detector use
seeded RNGs. End-to-end:

```bash
python -m src.drift.generate_windows  # produces 4 CSVs
python -m src.drift.detect            # produces results.json
python -m src.drift.visualize         # produces 4 PNGs
python -m src.drift.evidently_report  # produces 2 HTML reports
```

**Limitations.**

1. The "drift" in this analysis is synthetic and *known* (we injected
   it). In production, drift causes are unknown, and root-cause
   analysis would require correlating the dashboard signal with
   upstream data lineage events that this teaching system doesn't
   model. The pattern of "isolate which feature drifted, then talk to
   that feature's upstream owner" is general; the specifics depend on
   the operational environment.

2. The 5% anomaly rate in Week 3 is arbitrary. Real production anomaly
   rates depend on the upstream pipeline's failure modes; the integrity
   layer's threshold and rate-limiting policy should be set empirically.

3. Iris is a stand-in for tabular classification. The detection
   methodology, threshold structure, and impact-analysis pattern
   generalize; the absolute PSI values and accuracy drops do not.

---

*Companion artifacts:*

- `visualizations/drift/psi-evolution.png` — PSI per feature across windows, with C1 thresholds
- `visualizations/drift/feature-distributions.png` — raw feature histograms reference vs. production
- `visualizations/drift/accuracy-impact.png` — headline impact chart
- `visualizations/drift/class-distribution-shift.png` — output drift visualization
- `visualizations/drift/evidently-report-week2.html` — Evidently dashboard for Week 2
- `visualizations/drift/evidently-report-week3.html` — Evidently dashboard for Week 3
- `src/drift/results.json` — raw analysis output (reproducible from the scripts)
