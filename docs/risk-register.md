# Risk Register — Iris Classifier (Component 3)

This register documents identified risks for the deployed Iris
classifier, organized by the four categories the rubric specifies:
**bias, robustness, privacy, compliance**. Each risk is rated by
**likelihood** (Low / Medium / High) and **severity** (Low / Medium /
High) for *this teaching system*, with a separate column noting how
the rating would change if these patterns were applied to a
real-stakes production system.

The risk matrix in `docs/risk-matrix.md` (Component 5) extends this
register with a system-level view including retrieval risks,
tool-execution pathways, and the full likelihood × severity matrix.

## Bias

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| B-01 | Class imbalance bias — model learns to favor over-represented classes | Low | Low | High / High | Iris is balanced 50/50/50, so this is not an active concern. **Mitigation if reused:** stratified sampling at training time; per-class precision/recall in C1 dashboard; alert if any class falls below historical recall. |
| B-02 | Feature-engineering bias — interaction terms encode assumptions about feature relationships | Low | Low | Medium / Medium | The two engineered features (`sepal_l × sepal_w`, `petal_l × petal_w`) are documented in the model card. **Mitigation:** any change to the feature set requires a retrain + new C2 A/B test; the audit trail records feature-set changes as a separate event type. |
| B-03 | Subgroup performance bias — model performs differently across slices we don't measure | N/A (no demographic features) | N/A | High / High | Iris has no demographic features. **Mitigation if reused:** instrument per-subgroup accuracy as a Prometheus metric, with a dashboard panel per protected attribute; same alerting structure as the existing PSI panel but per subgroup. |
| B-04 | Reporting bias — model card under-states known limitations | Low | Medium | Medium / High | The model card explicitly names Iris as a stand-in and documents limitations. **Mitigation:** model card updates require a new audit-trail entry; reviewing the model card is part of the model promotion checklist. |

## Robustness

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| R-01 | Input drift silently degrades predictions | Medium | Medium | High / High | C1 dashboard exposes per-feature `iris_feature_drift_psi`; thresholds at 0.1 (yellow) and 0.25 (red) trigger investigation. C4 provides the offline batch-drift analysis that confirms what live PSI flagged. |
| R-02 | Adversarial inputs cause out-of-distribution predictions | Medium | Low (no consequence) | High / High | Integrity check rejects NaN/Inf at the API boundary; flags `|z|>5` post-scaling values via `iris_input_anomalies_total`. **Mitigation if reused:** add an out-of-distribution detector (e.g., isolation forest on training features) as a second-line filter. |
| R-03 | Model artifact corruption — pickle file changes silently | Low | High | Low / High | Model artifacts loaded once at service startup; SHA-256 of the artifact is logged on load. **Mitigation:** add a startup check that compares artifact hash against a known-good value recorded in the audit trail; fail-fast if mismatched. |
| R-04 | Scaler-model version drift — scaler params updated without retraining the model (or vice versa) | Low | High | Medium / High | Scaler and model are loaded from `models/` together; M3 lineage report ties them to the same MLflow run. **Mitigation:** the audit trail records both `model_path` and `scaler_path` together as a single deployment event; mismatched promotion is detectable on review. |
| R-05 | Cold-start latency spike causes timeout cascades upstream | Medium | Low | Medium / Medium | Observed in C1 dashboard testing: first prediction after service start is ~40 ms vs. ~5 ms steady state. **Mitigation:** add a synthetic warmup request at container startup; document the spike in dashboard interpretation so on-call engineers don't page on it. |
| R-06 | Metric-freshness false positive — gauges persist when data stops flowing, dashboard appears healthy | Medium | Medium | Medium / High | Identified during C1 testing — when traffic stopped, drift PSI gauges froze at last value rather than decaying. **Mitigation:** add a `up{job="iris-classifier"} == 0` alert in Prometheus; treat scrape staleness as a paging event regardless of metric values. |

## Privacy

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| P-01 | Training data contains PII | Low (Iris has none) | N/A | High / High | The Iris dataset is public-domain botanical measurements. **Mitigation if reused:** PII screen during data ingestion; documented retention period in M3 lineage; differential privacy or aggregation if any feature is sensitive. |
| P-02 | Inference logs leak input values to Prometheus through gauges | Medium | Low (Iris) | Low / High | C1 emits `iris_feature_value_mean` and `iris_feature_value_std` as gauges over a rolling window. For Iris these are harmless; on PII-bearing features, even aggregates can leak individual records if the window is small. **Mitigation if reused:** review every gauge for re-identification risk; enforce minimum window size; never emit raw per-request values. |
| P-03 | Audit trail leaks predictions or inputs | Low | Low (Iris) | Low / High | The audit trail records *events*, not per-request data — model promotions, configuration changes, monitoring incidents. **Mitigation if reused:** explicitly exclude per-request inputs/outputs from the audit trail; if those need logging, route them to a separately-secured datastore. |
| P-04 | Model memorization of training records | Low | Low (Iris) | Low / High | Iris training set is 150 records, far too small for memorization to be meaningful. **Mitigation if reused:** membership inference testing; differential privacy training where appropriate. |

## Compliance

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| C-01 | Lack of model documentation prevents audit | Medium | Medium | High / High | Model card (`docs/model-card.md`) covers the Mitchell et al. fields; lineage diagram traces data → deployment; audit trail logs lifecycle events. **Mitigation:** model card revision is a precondition for any model promotion (recorded in the promotion runbook). |
| C-02 | Untracked model promotions break reproducibility | Medium | High | High / High | The audit trail records every promotion with the M3 MLflow run ID, decision evidence, and authorizer. **Mitigation:** automated check that compares the running `iris_model_info` Prometheus gauge against the latest audit-trail entry; mismatch fails CI. |
| C-03 | Insufficient power on A/B test leads to ship/no-ship error (NIST AI RMF GOVERN-3) | Low | Medium | Medium / High | C2's experiment specification includes a defended power calculation (5,715 samples per arm for 1.0% MDE at α=0.05, power=0.80). **Mitigation:** any A/B test that ships requires the analysis JSON in the audit trail evidence column. |
| C-04 | No defined retention or deletion policy for monitoring data | Medium | Low (Iris) | Low / High | Prometheus is configured with `--storage.tsdb.retention.time=2d` in `monitoring-stack.yml`. **Mitigation if reused:** retention policy aligned with applicable regulation (e.g., GDPR Article 5 storage limitation); explicit deletion procedure in the runbook. |
| C-05 | Model decisions cannot be explained on request (NIST AI RMF MEASURE-2.7) | Low (Iris) | Low (Iris) | Medium / High | RandomForestClassifier supports `feature_importances_` and is locally explainable via tools like SHAP/LIME. **Mitigation if reused:** add an explanation endpoint that returns top-k contributing features per prediction; record explanation availability in the model card. |

## Severity / likelihood scoring rubric

For consistency across this register and the risk matrix in C5:

| Likelihood | Definition |
|---|---|
| Low | Has not happened; would require a specific failure mode |
| Medium | Plausible within a quarter of operation |
| High | Has happened during development or expected to recur |

| Severity | Definition |
|---|---|
| Low | Operator notices but no user impact |
| Medium | Some users see degraded service; recoverable within hours |
| High | User-facing outage, regulatory exposure, or data loss |

## Cross-references

- Items B-01 through B-04 connect to the **Ethical considerations**
  section of the model card.
- Items R-01 and R-06 are actively monitored by the C1 dashboard.
- Item R-01 is the topic of C4's offline drift analysis.
- Item C-02 is enforced by the audit trail (`logs/audit-trail.json`)
  and the `iris_model_info` Prometheus gauge.
- Item C-03 is the rubric criterion that C2's experiment
  specification explicitly addresses.
- The system-level extension of this register — including retrieval
  and tool-execution risks not relevant to this single-model service
  — appears in `docs/risk-matrix.md` (C5).
