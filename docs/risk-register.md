# Risk Register — Iris Classifier (Component 3)

This register lists the risks I've identified for the deployed Iris
classifier, organized by the four categories the rubric specifies:
**bias, robustness, privacy, compliance**. Each risk gets a
**likelihood** (Low / Medium / High) and **severity** (Low / Medium /
High) rating *for this teaching system*, with a separate column noting
how the rating would change if the same patterns were applied to a
real-stakes production system.

The system-level risk matrix in `docs/risk-matrix.md` (Component 5)
extends this register with retrieval risks, tool-execution pathways,
and the full likelihood × severity scoring.

## Bias

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| B-01 | Class imbalance bias — model favors over-represented classes | Low | Low | High / High | Iris is balanced 50/50/50, so this isn't really a concern here. **If reused:** stratified sampling at training time; per-class precision/recall in the C1 dashboard; alert if any class drops below its historical recall. |
| B-02 | Feature-engineering bias — interaction terms encode assumptions about how features relate | Low | Low | Medium / Medium | The two engineered features (`sepal_l × sepal_w`, `petal_l × petal_w`) are documented in the model card. **Mitigation:** any change to the feature set means a retrain plus a new C2 A/B test, and the audit trail records feature-set changes as their own event type. |
| B-03 | Subgroup performance bias — the model performs differently across slices we don't measure | N/A (no demographic features) | N/A | High / High | Iris doesn't have demographic features. **If reused:** track per-subgroup accuracy as a Prometheus metric, with a dashboard panel per protected attribute. Same alerting structure as the existing PSI panel, just sliced by subgroup. |
| B-04 | Reporting bias — the model card under-states known limitations | Low | Medium | Medium / High | The model card explicitly names Iris as a stand-in and documents limitations. **Mitigation:** model card updates need a new audit-trail entry, and reviewing the model card is part of the model promotion checklist. |

## Robustness

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| R-01 | Input drift quietly degrades predictions | Medium | Medium | High / High | The C1 dashboard exposes per-feature `iris_feature_drift_psi`; thresholds at 0.1 (yellow) and 0.25 (red) trigger investigation. C4 provides the offline batch-drift analysis that confirms what live PSI flagged. |
| R-02 | Adversarial inputs cause out-of-distribution predictions | Medium | Low (no consequence here) | High / High | The integrity check rejects NaN/Inf at the API boundary and flags `|z|>5` post-scaling values via `iris_input_anomalies_total`. **If reused:** add an out-of-distribution detector (something like an isolation forest on training features) as a second-line filter. |
| R-03 | Model artifact corruption — pickle file changes silently | Low | High | Low / High | Model artifacts get loaded once at service startup; the SHA-256 of the artifact is logged on load. **Mitigation:** add a startup check that compares the artifact hash against a known-good value recorded in the audit trail; fail-fast if they don't match. |
| R-04 | Scaler-model version drift — scaler params updated without retraining the model (or vice versa) | Low | High | Medium / High | The scaler and model are loaded together from `models/`, and the M3 lineage report ties them to the same MLflow run. **Mitigation:** the audit trail records both `model_path` and `scaler_path` together as a single deployment event, so a mismatched promotion is detectable on review. |
| R-05 | Cold-start latency spike causes timeout cascades upstream | Medium | Low | Medium / Medium | I saw this during C1 dashboard testing — the first prediction after the service starts is around 40 ms vs. ~5 ms steady state. **Mitigation:** add a synthetic warmup request at container startup, and document the spike in the dashboard interpretation so on-call engineers don't page on it. |
| R-06 | Metric-freshness false positive — gauges keep their last value when data stops flowing, so the dashboard looks healthy when it isn't | Medium | Medium | Medium / High | I caught this during C1 testing — when traffic stopped, the drift PSI gauges froze at their last value instead of decaying. **Mitigation:** add a `up{job="iris-classifier"} == 0` alert in Prometheus and treat scrape staleness as a paging event regardless of what the metric values say. |

## Privacy

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| P-01 | Training data contains PII | Low (Iris has none) | N/A | High / High | The Iris dataset is public-domain botanical measurements. **If reused:** PII screen during data ingestion, documented retention period in M3 lineage, and differential privacy or aggregation if any feature is sensitive. |
| P-02 | Inference logs leak input values to Prometheus through gauges | Medium | Low (Iris) | Low / High | C1 emits `iris_feature_value_mean` and `iris_feature_value_std` as gauges over a rolling window. For Iris these are harmless, but on PII-bearing features even aggregates can leak individual records if the window is small. **If reused:** review every gauge for re-identification risk, enforce a minimum window size, and never emit raw per-request values. |
| P-03 | Audit trail leaks predictions or inputs | Low | Low (Iris) | Low / High | The audit trail records *events*, not per-request data — model promotions, configuration changes, monitoring incidents. **If reused:** explicitly exclude per-request inputs/outputs from the audit trail; if those need logging, route them to a separately-secured datastore. |
| P-04 | Model memorization of training records | Low | Low (Iris) | Low / High | The Iris training set is 150 records, way too small for memorization to actually matter. **If reused:** membership inference testing and differential privacy training where appropriate. |

## Compliance

| ID | Risk | Likelihood | Severity | If real-stakes? | Mitigation |
|---|---|---|---|---|---|
| C-01 | Lack of model documentation prevents audit | Medium | Medium | High / High | The model card (`docs/model-card.md`) covers the Mitchell et al. fields, the lineage diagram traces data → deployment, and the audit trail logs lifecycle events. **Mitigation:** model card revision is a precondition for any model promotion (recorded in the promotion runbook). |
| C-02 | Untracked model promotions break reproducibility | Medium | High | High / High | The audit trail records every promotion with the M3 MLflow run ID, decision evidence, and authorizer. **Mitigation:** an automated check that compares the running `iris_model_info` Prometheus gauge against the latest audit-trail entry — a mismatch fails CI. |
| C-03 | Insufficient power on an A/B test leads to a ship/no-ship error (NIST AI RMF GOVERN-3) | Low | Medium | Medium / High | C2's experiment specification includes a defended power calculation (5,715 samples per arm for a 1.0% MDE at α=0.05, power=0.80). **Mitigation:** any A/B test that ships needs the analysis JSON in the audit trail evidence column. |
| C-04 | No defined retention or deletion policy for monitoring data | Medium | Low (Iris) | Low / High | Prometheus is configured with `--storage.tsdb.retention.time=2d` in `monitoring-stack.yml`. **If reused:** retention policy aligned with whatever regulation applies (e.g., GDPR Article 5 storage limitation), with an explicit deletion procedure in the runbook. |
| C-05 | Model decisions can't be explained on request (NIST AI RMF MEASURE-2.7) | Low (Iris) | Low (Iris) | Medium / High | RandomForestClassifier supports `feature_importances_` and is locally explainable via SHAP/LIME. **If reused:** add an explanation endpoint that returns the top-k contributing features per prediction, and record explanation availability in the model card. |

## Severity / likelihood scoring rubric

For consistency across this register and the risk matrix in C5:

| Likelihood | What it means |
|---|---|
| Low | Hasn't happened; would need a specific failure mode |
| Medium | Plausible within a quarter of operation |
| High | Has happened during development, or expected to recur |

| Severity | What it means |
|---|---|
| Low | The operator notices but no user is affected |
| Medium | Some users see degraded service; recoverable within hours |
| High | User-facing outage, regulatory exposure, or data loss |

