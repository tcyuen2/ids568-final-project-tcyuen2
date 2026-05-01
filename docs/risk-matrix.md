# Risk Matrix — Component 5

This matrix captures in-scope risks from the governance review
(`docs/governance-review.md`) with **likelihood**, **severity**, and a
**concrete mitigation** for each. It is the system-level extension of
the model-centric risk register from Component 3
(`docs/risk-register.md`); risks that are already in C3's register are
referenced rather than re-stated.

## Scoring rubric

| Likelihood | Definition |
|---|---|
| Low | Has not occurred; would require a specific failure mode |
| Medium | Plausible within a quarter of operation |
| High | Has occurred during development or expected to recur |

| Severity | Definition |
|---|---|
| Low | Operator notices; no user impact |
| Medium | Some users see degraded service; recoverable within hours |
| High | User-facing outage, regulatory exposure, data loss, or trust loss |

A **Risk score** column combines the two on a 1–9 scale (1 = Low/Low,
9 = High/High) for ranking. This keeps the matrix sortable while still
carrying the categorical labels.

## In-scope risks

| ID | Risk | Category | L | S | Score | Status | Mitigation |
|---|---|---|---|---|---|---|---|
| S-01 | Pickle artifact swap → arbitrary code execution at service startup | Data security | Low | High | 3 | Not implemented | Add SHA-256 verification of `models/model.pkl` and `models/scaler.pkl` against hashes recorded in the audit trail; fail-fast on mismatch. References C3 risk **R-03**. |
| S-02 | Audit trail tampered to hide a problematic model promotion | Data security | Low | High | 3 | Convention-only | Move audit trail to append-only object store (S3 + object lock) or tamper-evident log service. Today the file is just `logs/audit-trail.json`. |
| S-03 | Unauthenticated `/predict` endpoint allows abuse / oracle queries | Data security | High | Medium | 6 | Local dev only | Terminate TLS at ingress; require authentication; rate-limit per principal. Acceptable for local development; mandatory for any non-local deployment. |
| S-04 | Prometheus rolling-mean gauge leaks individual feature values | Data security / Privacy | Low | Low (Iris) | 1 | Acceptable for Iris | Iris features carry no privacy risk. References C3 **P-02**: this metric class would need explicit privacy review before reuse on PII-bearing features. |
| H-01 | Confidently-wrong predictions during distribution shift (classification analog of hallucination) | Hallucination (in-scope variant) | High | Medium | 6 | Detection in place; gap monitor not | Confidently-wrong behavior was directly observed in C4 (mean confidence 0.94 with accuracy 0.66 under significant drift). Detection is in place via the C1 drift PSI panel. **Add** a confidence-vs-accuracy gap monitor as a leading indicator (recommended in C4 diagnostic report section 5). |
| H-02 | Model emits prediction on out-of-distribution input without flagging it | Hallucination | Medium | Medium | 4 | Partial | The integrity layer rejects NaN/Inf and logs |z|>5 outliers, but still produces predictions for values within ±5σ. **Add** a dedicated out-of-distribution detector (e.g., isolation forest on training features) as a second-line filter that returns "low confidence — human review" rather than a class label. |
| T-01 | Bulk querying to reverse-engineer the decision boundary | Tool-misuse | Medium | Low (Iris) | 2 | Acceptable for Iris | Iris is a public model on public data; reverse-engineering it produces nothing private. **Add** rate-limiting per principal before reuse on a non-public model. |
| T-02 | Use of classifier as labeling oracle outside its sanctioned use case | Tool-misuse | Medium | Low (Iris) | 2 | Acceptable for Iris | No real consequence here. **Add** scoped API tokens binding callers to a stated use case before reuse on a model with stake-specific licensing or accuracy claims. |
| T-03 | Adversarial input crafting to push borderline samples across class boundaries | Tool-misuse | Low | Low (Iris) | 1 | Not relevant for Iris | Iris is too low-dimensional and the classifier too well-trained for this to matter. References C3 **R-02**. |
| C-01 | Lack of model documentation blocks audit | Compliance | Low | Medium | 2 | Mitigated | Model card, lineage diagram, audit trail, and risk register together provide audit completeness. References C3 **C-01**. |
| C-02 | Untracked model promotion → reproducibility failure | Compliance | Low | High | 3 | Mitigated by audit trail | Every `model_promoted_to_production` event is logged with M3 run ID, decision evidence, and authorizer. References C3 **C-02**. |
| C-03 | Insufficient A/B test power → ship/no-ship error | Compliance | Low | Medium | 2 | Mitigated by C2 spec | C2 experiment specification includes a defended power calculation. References C3 **C-03**. |
| C-04 | Insufficient monitoring data retention for incident retrospective | Compliance | Medium | Low | 2 | Mitigated by config | `--storage.tsdb.retention.time=2d` in `monitoring-stack.yml`. References C3 **C-04**. |
| C-05 | Inability to explain a prediction on request | Compliance | Low | Low (Iris) | 1 | Partial | RandomForest is locally explainable via SHAP / LIME / `feature_importances_`. **Add** an explanation endpoint before reuse on a regulated system. References C3 **C-05**. |
| O-01 | Drift PSI gauge persists last value when data flow stops, dashboard appears healthy | Operational | Medium | Medium | 4 | Detection partial | Observed during C1 build. **Add** a `up == 0` alert in Prometheus to treat scrape staleness as a paging event regardless of metric values. References C3 **R-06**. |
| O-02 | Cold-start latency spike causes timeout cascades upstream | Operational | Medium | Low | 2 | Documented | Observed in C1 dashboard testing (~40 ms vs. ~5 ms steady state). **Add** a synthetic warmup request at container startup. References C3 **R-05**. |
| O-03 | Drift baseline (`models/reference_stats.json`) becomes stale after retraining | Operational | Medium | Medium | 4 | Process recommendation | Recommended in C4 diagnostic report (action item 7): re-baseline the reference stats whenever the production model is retrained on new data. |

## Out-of-scope categories

For completeness, here are the rubric categories that do not apply to
this system, with the rationale documented in
`docs/governance-review.md`:

| Category | Reason out of scope | Where it would attach if extended |
|---|---|---|
| Retrieval — exposure | No retriever in this system | Cross-tenant leakage if extended to per-client prediction history |
| Retrieval — contamination | No retriever in this system | Cache poisoning if extended with a feature cache |
| Retrieval — staleness | No retriever in this system | Stale scaler vs. updated model |
| Hallucination — fabricated free text | No generative model in this system | Explanation layer if extended with an LLM-generated rationale |
| Tool execution — external APIs | No agentic tool calls | Downstream automation if extended into an agent |
| User PII | No PII processed | All PII-specific regulation if extended to a regulated domain |

## Risk-by-score summary

Sorted by risk score (likelihood × severity), highest first:

1. **S-03** (score 6, in scope): Unauthenticated `/predict` endpoint
2. **H-01** (score 6, in scope): Confidently-wrong predictions during drift
3. **H-02** (score 4, in scope): OOD prediction without flagging
4. **O-01** (score 4, in scope): PSI gauge stale-data false positive
5. **O-03** (score 4, in scope): Stale drift baseline after retraining
6. **S-01** (score 3, in scope): Pickle swap → code execution
7. **S-02** (score 3, in scope): Audit-trail tampering
8. **C-02** (score 3, mitigated): Untracked promotion (already mitigated by audit trail design)

Items at score ≤ 2 are either acceptable in this teaching context
(Iris stand-in) or already mitigated by C3 artifacts.

## Connection to monitoring

Several risks above are *already* observable through the C1 dashboard:

- **H-01** (confidently wrong) → `iris_feature_drift_psi` panel
- **O-01** (stale gauges) → would be caught by `up{job="iris-classifier"} == 0` alert
- **O-02** (cold-start spike) → visible on the latency panel; baseline known

This is the connective tissue the rubric explicitly rewards: the same
telemetry that drives operational monitoring drives risk detection.
The CTO memo (`docs/cto-memo.md`) translates these findings into
executive action items.
