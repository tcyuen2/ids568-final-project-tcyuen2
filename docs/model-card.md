# Model Card — Iris Classifier (Final Project Build)

This model card follows the structure proposed by Mitchell et al. (2019,
*Model Cards for Model Reporting*) and accompanies the Iris classifier
deployed in this repository's monitoring stack (Component 1) and
validated by the A/B test in Component 2.

## Model details

| Field | Value |
|---|---|
| Service name | `iris-classifier` |
| Active model variant | `baseline_run` (after C2 A/B promotion) |
| Previous variant (control) | `high_min_split` (was active during C1 build) |
| Model architecture | scikit-learn `RandomForestClassifier` |
| Active hyperparameters | `n_estimators=50`, `max_depth=5`, `min_samples_split=2` |
| Number of features | 6 (4 raw + 2 engineered interactions) |
| Number of classes | 3 (`setosa`, `versicolor`, `virginica`) |
| Model framework version | scikit-learn 1.3.2 |
| Serialization | Python pickle (.pkl) |
| Active artifact path | `models/model.pkl` |
| Lineage — M3 MLflow run ID (active) | `0ab6a6fa80f949d5b917115218e126c4` |
| Lineage — M3 MLflow run ID (previous) | `6e01be84da0e47d68b1fc5caa9749e40` |
| Training environment | Apache Airflow 2.7.3 + MLflow 2.9.2 (M3 stack) |
| Serving environment | FastAPI 0.115 + Uvicorn 0.30.6 + Docker (C1 stack) |
| Production monitoring | Prometheus 2.54 + Grafana 11.2 (C1 stack) |
| License | Educational use; no proprietary data |
| Owner | tcyuen2 (IDS568 student) |

The active variant changed from `high_min_split` to `baseline_run`
after the A/B test described in Component 2. See
`logs/audit-trail.json` for the full promotion record.

## Intended use

**Intended primary use cases.** This model exists as a **stand-in for
tabular classification** in an MLOps teaching context. It demonstrates
the operational practices — instrumented serving, drift detection,
A/B testing, governance documentation, risk assessment — that
generalize to production tabular ML systems.

**Intended users.** Course graders, the model's author, and any
engineer cloning this repository to study or extend the operational
patterns.

**Out-of-scope use cases.** This model **must not** be used for:

- Any real-world botanical identification task. The training set is
  150 samples of three Iris species published in 1936; it is not a
  general-purpose plant classifier.
- Any decision with consequences for people (loans, employment,
  health). Iris carries no such concerns; deploying its operational
  patterns to a high-stakes domain would require additional governance.
- Drawing conclusions about "production ML at scale." Iris is a small,
  clean, balanced dataset. Real production data has class imbalance,
  noisy labels, and adversarial inputs that this system has not been
  exercised against.

## Factors

**Relevant factors that could affect performance:**

- Sample provenance — the synthetic traffic generator samples from
  per-class normals fitted to the original 1936 dataset. Performance
  is measured against this synthetic distribution; it would degrade
  on samples drawn from any other distribution (e.g., real flower
  measurements with seasonal or geographic variation).
- Input scaling — predictions depend on `StandardScaler` parameters
  fit during M3 training. If the bundled scaler is replaced with a
  different one, predictions will silently miscompute.
- Feature engineering — the model expects exactly the two interaction
  features used during M3 training (`sepal_length × sepal_width`,
  `petal_length × petal_width`). Adding, removing, or reordering
  features will fail the input-shape check at inference time.

**Evaluation factors not exercised:**

- Adversarial inputs (other than the simple `|z| > 5` integrity
  check in C1)
- Demographic subgroups (Iris has no demographic features)
- Geographic or temporal subgroups

## Training data

**Source:** The classic Iris dataset (Fisher 1936), accessed via
`sklearn.datasets.load_iris()`. 150 samples, balanced 50/50/50 across
three species: *Iris setosa*, *Iris versicolor*, *Iris virginica*.

**Features in raw data (4):**
- `sepal_length` (cm)
- `sepal_width` (cm)
- `petal_length` (cm)
- `petal_width` (cm)

**Engineered features added during M3 preprocessing (2):**
- `sepal_l_x_sepal_w` = `sepal_length × sepal_width`
- `petal_l_x_petal_w` = `petal_length × petal_width`

**Preprocessing:** `StandardScaler` fit on the training split
(stratified 80/20 train/test). The fit scaler is bundled at
`models/scaler.pkl` and is the only one the serving service knows.

**Data lineage:** SHA-256 hashes of the loaded data are recorded
in M3's `lineage_report.md`. These hashes form the upstream anchor
for every prediction this service makes.

**No PII, no proprietary data.** The Iris dataset is public-domain
botanical measurements; nothing is private about it.

## Evaluation data

**Source:** The 30-sample test split (20% of the 150-sample dataset)
held out during M3 training. Same source as training data; same
preprocessing.

**Limitation that drove C2:** A 30-sample test set has wide confidence
intervals. The 3.4-percentage-point offline accuracy gap between
`high_min_split` and `baseline_run` reported on this test set was not
statistically distinguishable from noise (this was confirmed in C2
with n≈5,700 per arm).

**Live-traffic evaluation:** The C2 simulation provides a much larger
evaluation under synthetic-traffic conditions:

- Variant A (`high_min_split`): 5,617/5,750 correct = 0.9769 accuracy
- Variant B (`baseline_run`): 5,573/5,680 correct = 0.9812 accuracy
- 95% CI on difference (A − B): [−0.0095, +0.0010]

## Quantitative analyses

**M3 offline metrics for the active variant (`baseline_run`):**

| Metric | Value |
|---|---|
| Accuracy | 0.933 |
| Precision (macro) | 0.933 |
| Recall (macro) | 0.933 |
| F1 (macro) | 0.933 |
| AUC (one-vs-rest macro) | 0.993 |

These are the offline numbers that motivated the A/B test, not the
final word on production performance.

**C2 simulation metrics for the active variant:**

| Metric | Value |
|---|---|
| Accuracy on synthetic traffic | 0.9812 (n=5,680) |
| Difference vs. previous variant | −0.0043 (in B's favor) |
| Two-sided p-value | 0.110 |
| Decision | SHIP_B (non-inferior to A by 3% margin) |

**Per-class breakdown (M3 evaluation):** Iris is balanced, and the
test set is stratified, so per-class performance is approximately
the headline accuracy. No subgroup analysis applies (Iris has no
demographic features).

## Ethical considerations

**Direct ethical exposure: minimal.** The model classifies plant
species from physical measurements. There are no individuals, no
demographic features, and no consequential decisions involved in
its intended use.

**Indirect ethical considerations the *operational pattern* must
respect when reused on a different model/dataset:**

- **Bias.** A real production model on demographic features could
  encode bias. The risk register (`docs/risk-register.md`) documents
  the audit pathway: per-subgroup performance metrics, drift
  detection per subgroup, and the human-in-the-loop escalation
  channel for high-uncertainty predictions.
- **Privacy.** The current system holds no PII, but the Prometheus
  metrics emitted include feature *values* through gauges (e.g.,
  `iris_feature_value_mean`). On a real PII-bearing system, this
  channel would need explicit privacy review — gauges of mean and
  std are usually fine, but rolling-window contents could leak
  individual records if mishandled.
- **Transparency.** This card, the lineage diagram, and the audit
  trail together make every prediction traceable to a training run,
  a serving version, a deployment event, and an authorizer.

## Caveats and recommendations

1. **Iris is a teaching dataset.** Every claim about "production
   readiness" in this repository should be read as "production
   patterns demonstrated against a teaching system." Real
   production deployment requires retraining on real data,
   subgroup analysis, and a human review process not implemented
   here.

2. **The active model has higher variance than its predecessor.**
   `baseline_run` uses 50 trees of depth 5 — a smaller forest than
   the previous variant. While the C2 simulation showed equivalent
   accuracy on synthetic traffic, the smaller ensemble has wider
   per-prediction confidence variance. Watch the
   `iris_prediction_confidence` panel in the C1 dashboard for
   sustained shifts after the promotion.

3. **Drift baselines are not zero.** The synthetic traffic generator
   does not exactly match the M3 training distribution, so the
   C1 dashboard's `iris_feature_drift_psi` will sit between 0.05
   and 0.2 even under "normal" traffic. Alert thresholds in the
   dashboard interpretation document (Component 1) account for this
   baseline.

4. **No retraining loop.** This system uses a frozen artifact from
   M3 and has no automated retraining trigger. If C4 drift detection
   identifies persistent drift, retraining is a manual operation:
   re-run the M3 Airflow DAG, register the new MLflow run, repeat
   the C2 A/B test against the current production model, and follow
   the promotion procedure recorded in the audit trail.

## Versioning and update policy

- Model artifact changes are recorded in `logs/audit-trail.json`.
- This card is updated whenever the active variant changes or when
  any quantitative metric is re-measured.
- The C1 dashboard's `iris_model_info` Prometheus gauge carries
  the active variant name and M3 run ID as labels — this is the
  ground truth for "what is currently serving" at any moment.

---

*Last updated to reflect the C2 A/B test outcome (SHIP_B decision)
and Component 3 deliverables. See* `docs/recommendation-memo.md` *for
the underlying decision and* `logs/audit-trail.json` *for the
promotion event.*
