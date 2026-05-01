# Memorandum

**To:** Chief Technology Officer
**From:** [Your name], MLOps team
**Date:** [Submission date]
**Re:** Iris Classifier Service — operational readiness, risk posture, and recommendations
**Priority:** Routine review (no active incident)

---

## Executive summary

The Iris classifier service has reached operational maturity across
five dimensions: instrumented serving, statistically-defended model
promotion, governance documentation, drift detection, and risk
assessment. **No active risks require immediate action.** Three
follow-up items are recommended for the next quarter to close gaps
between the current implementation and what would be required for
regulated production use.

The system itself is a teaching deployment built on the public Iris
botanical dataset. Its production-readiness story is therefore
**operational, not domain-specific**: the patterns demonstrated
generalize directly to tabular ML classification at scale, but the
absolute risk picture would change for any system handling personal
data or stake-bearing decisions. Section 4 below lists the specific
extensions that would be needed.

## What was built

| Component | Outcome |
|---|---|
| Production monitoring (C1) | FastAPI service with 12 Prometheus metrics, 6-panel Grafana dashboard, Docker Compose stack. Drift, latency, and integrity all observable in real time. |
| A/B test (C2) | Statistically defended promotion of `baseline_run` over `high_min_split`. The 3.4-point offline accuracy gap proved to be small-sample noise — a real finding, not a contrived result. |
| Governance (C3) | Model card, risk register, lineage diagram, append-only audit trail. The promotion decision from C2 is recorded in the audit trail with the analysis output as evidence. |
| Drift detection (C4) | Offline confirmation of what C1's live panel surfaces. Quantified: a 20% petal-channel drift produces a 19% accuracy drop. |
| Risk assessment (C5) | This memo and its supporting artifacts. |

## Three findings worth your attention

### 1. The model is "confidently wrong" under distribution shift

The drift analysis revealed a specific failure mode: under a 1.5×
petal-measurement shift, accuracy collapses from 0.99 to 0.66 while
mean prediction confidence stays at 0.94. **The model does not know
when it doesn't know.** This is not a bug in our model — it's a
known property of tree ensembles on out-of-distribution inputs. It
matters because:

- A naïve confidence-only monitor would miss this entirely.
- Our drift-PSI panel does catch it, because it monitors *input
  distributions* rather than *output confidence*.
- The recommended follow-up is to add a confidence-vs-accuracy gap
  monitor as a leading indicator. Cost: small (one Prometheus
  metric, one Grafana panel).

### 2. The 0.25 PSI alert threshold is too coarse for this model

Industry-standard PSI thresholds (0.10 / 0.25) are based on credit
risk literature. They work as a starting point but aren't tuned to
*this* model's sensitivity profile. C4 found that petal-feature PSI
above 0.25 produces severe accuracy loss, while sepal-feature PSI
above 0.25 has limited accuracy impact. **Per-feature alert
thresholds**, calibrated to per-feature accuracy sensitivity, would
reduce both false alarms (sepal noise) and missed alerts (low petal
PSI causing real damage).

### 3. The audit trail is append-only by convention, not enforcement

Today, `logs/audit-trail.json` is a regular file. A motivated bad
actor with write access could rewrite history. For the educational
context this is fine; **for any regulated deployment it is not.**
The fix is to move the audit trail to write-once storage (S3 with
object lock, or a tamper-evident log service). Cost: small at this
volume; non-negotiable at any reasonable stake.

## Recommended actions

In priority order:

| # | Action | Owner | Effort | Why |
|---|---|---|---|---|
| 1 | Add confidence-vs-accuracy gap monitor | Monitoring lead | 1 day | Closes the "confidently wrong" detection gap (Finding 1) |
| 2 | Calibrate per-feature PSI alert thresholds | C4 lead | 2 days | Reduces false alarms; catches early petal drift (Finding 2) |
| 3 | Move audit trail to append-only storage | SRE lead | 3 days | Required precondition for any regulated reuse (Finding 3) |
| 4 | Add SHA-256 verification of model artifacts at startup | Service owner | 1 day | Closes the pickle-swap code-execution risk (S-01 in risk matrix) |
| 5 | Re-baseline drift reference whenever model is retrained | Process change | n/a | Prevents stale-baseline false positives after retraining (O-03) |

Items 1, 2, and 5 are essentially free and produce real risk
reduction. Items 3 and 4 are precondition items for any reuse on a
regulated deployment.

## What this system does *not* tell us

This review covers what was built and how it behaves. It does **not**
cover:

- **Real-stakes regulatory exposure.** Iris carries no PII and no
  decision consequence. A loan-approval or medical-screening reuse
  of these patterns would require additional governance not
  prototyped here. The risk matrix's "Out of scope" section enumerates
  what would attach.
- **Adversarial robustness.** Iris is too low-dimensional for
  meaningful adversarial input research; the integrity layer rejects
  obvious anomalies but a real adversarial-relevant deployment
  needs an out-of-distribution detector as a second line of defense.
- **Multi-tenant operation.** This service is single-tenant. Reuse
  by multiple downstream consumers would require principal-bound
  authorization, rate-limiting per principal, and per-tenant metrics.

## Bottom line

The system is fit for its declared purpose: an educational MLOps
deployment that demonstrates production-grade operational patterns.
Three small, cheap follow-ups would meaningfully improve the
operational risk posture even within the educational scope. Three
larger items would be required preconditions for any reuse on a
regulated or higher-stakes domain. None of these items represents
an active incident or a deployment-blocking concern at the current
scope.

---

**Supporting documents:**
- `docs/system-boundary-diagram.png` — what's in the system, what isn't
- `docs/governance-review.md` — full structured review by category
- `docs/risk-matrix.md` — likelihood × severity scoring with mitigations
- `docs/model-card.md` — model details and intended use
- `docs/risk-register.md` — model-level risk inventory (C3)
- `docs/drift-diagnostic-report.md` — drift impact analysis (C4)
- `docs/recommendation-memo.md` — the A/B promotion decision (C2)
- `logs/audit-trail.json` — append-only event log
