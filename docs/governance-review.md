# Governance Review — Component 5

This is a system-level governance review of the deployed Iris classifier
service, organized around the five rubric categories: data security,
retrieval risks, hallucination risks, tool-misuse pathways, and
compliance concerns.

## A note on scope

The rubric example for this component frames the system as
"retriever → LLM → tool use → final output" — a typical agentic/RAG
pattern from Module 7. **The system under review is not that shape.**
It is a tabular RandomForest classifier behind a FastAPI service, with
no LLM, no retriever, no vector store, and no agentic tool execution.

This review takes that mismatch head-on rather than papering over it.
For each rubric category, this document states clearly:

1. **What is in scope** — the real risks present in the deployed system.
2. **What is out of scope, and why** — the rubric categories that don't
   map (with the reasoning, not just an "N/A").
3. **How the category would apply if extended** — a brief note on how
   each out-of-scope category would attach if the system were extended
   toward more typical Module 7 patterns.

The system boundary is depicted in `docs/system-boundary-diagram.png`,
which explicitly marks the four out-of-scope domains (LLM, retriever,
tool execution, user PII store).

## 1. Data security

### In scope

**Inbound data: feature values from upstream callers.** The `/predict`
endpoint accepts JSON over HTTP. In the local development stack the
service is unauthenticated; in any non-trivial deployment, this is the
single highest-priority hardening task: terminate TLS at an ingress,
require authentication, rate-limit per principal.

**Model artifacts at rest.** `models/model.pkl` and
`models/scaler.pkl` are deserialized at startup. Python pickle is a
known code-execution vector — a swapped pickle file becomes
arbitrary code execution inside the service process. This is **R-03**
in the C3 risk register; the mitigation (artifact-hash check at load,
fail-fast on mismatch) is recommended but not yet implemented.

**Audit-trail integrity.** `logs/audit-trail.json` is append-only by
convention rather than by enforcement (it's a regular file). On a real
deployment, write-once storage (S3 with object lock, or a tamper-evident
log service) would be the right primitive.

**Monitoring data.** Prometheus scrapes `/metrics` on a 5-second
interval. The metrics include rolling per-feature mean/std gauges,
which in the Iris context are harmless aggregates but on a real
PII-bearing system would need privacy review. See **P-02** in the
C3 risk register.

### Out of scope

**No user PII or credentials are processed by this service.** Iris is
public-domain botanical measurements; nothing about a flower's sepal
length is sensitive. This eliminates the entire class of risks around
encryption-at-rest of personal data, GDPR data subject rights, and
right-to-erasure procedures that would dominate a real-stakes
deployment.

### How this category would apply if extended

If the service were repurposed for any PII-bearing classification
(loan approval, medical screening), data security expands sharply:
input encryption in transit and at rest, secrets management for the
model artifact's lineage chain, retention policies aligned with
applicable regulation, and a documented data-flow review. The
existing audit-trail event types would need to be extended with
data-access events.

## 2. Retrieval risks

### In scope

**None.** The system performs no retrieval. There is no vector store,
no document index, no RAG pipeline. The model receives features
directly from the API caller and produces predictions deterministically.

### Out of scope, and why

The three retrieval-specific risks named by the rubric — **exposure
of retrieved data**, **prompt-context contamination**, **stale
knowledge in retrieved documents** — require a retriever to apply.
This system has none, so the categories cannot manifest.

This is not a limitation to lament; for an MLOps risk review of *this*
system, retrieval risks are the wrong question. The right question
("what happens if a single feature's upstream source corrupts?") is
analyzed under "Data security" above and in the C3 risk register
(**R-01**, **R-06**).

### How this category would apply if extended

If the service were extended to retrieve historical predictions for
the same client (e.g., "show this client's last 5 predictions before
making a new one"), retrieval risks would attach as follows:

- *Exposure*: cross-tenant leakage if the retrieval scope isn't bound
  to the calling principal — the standard authorization problem at
  the data-access layer.
- *Contamination*: a malicious historical prediction (or a corrupted
  cache entry) influencing the new prediction's pre-processing — the
  classic cache-poisoning pattern, mitigated with cache integrity
  checks and bounded TTLs.
- *Staleness*: the model is updated but the cached features are still
  scaled by the old StandardScaler — same shape as the
  scaler-vs-model-version-drift risk (**R-04** in C3) extended to
  cached data.

## 3. Hallucination risks

### In scope

**Overconfident wrong predictions.** The C4 drift analysis directly
surfaced this: under significant drift, the model retains 0.94 mean
prediction confidence while accuracy collapses to 0.66 — a 28-point
"confidently wrong" gap. This is a *classification analog* of
hallucination: the model emits a structured answer with apparent
certainty when it is in fact operating outside its training
distribution.

This risk is real and present in the deployed system. The mitigation
is in place: the C1 dashboard's `iris_feature_drift_psi` panel makes
out-of-distribution traffic visible regardless of the model's
confidence about its predictions. The C4 diagnostic report
(`docs/drift-diagnostic-report.md`) recommends adding a
confidence-vs-accuracy gap monitor as a follow-up.

### Out of scope, and why

The rubric's hallucination category is canonically about **LLM
free-text generation** — a generative model fabricating facts,
attributions, or citations. This system generates no free text. Its
output is constrained to a 3-element class probability vector;
fabrication of new content categories is mechanically impossible.

### How this category would apply if extended

If the service were extended to produce a **textual explanation** of
each prediction via an LLM (e.g., "this flower was classified as
*virginica* because…"), then classical LLM hallucination risks would
attach to the explanation layer: incorrect botanical facts, confident
references to nonexistent studies, and ungrounded claims about
feature importances that don't actually appear in the underlying
model. The mitigation would be RAG-grounding the explanations against
a verified botanical knowledge base and refusing-to-answer when the
retrieval is empty.

## 4. Tool-misuse pathways

### In scope

**The classifier itself can be considered a "tool" callable by an
upstream caller.** Misuse pathways for this tool include:

- **Bulk-querying** to reverse-engineer the decision boundary
  (e.g., to learn what features the model uses, then craft adversarial
  inputs). Mitigation: rate-limit per principal — not currently
  implemented, but called out in **R-02** of the C3 risk register.
- **Repeated probing for class assignment** to use the classifier as
  a labeling oracle for a downstream system the model wasn't
  authorized to support. Mitigation: scoped API tokens that bind
  callers to their stated use case.
- **Adversarial input crafting** to push borderline samples across
  class boundaries. Iris is too low-dimensional and the classifier
  too well-trained for this to matter in practice; on a real model
  with adversarial-relevant stakes, an out-of-distribution detector
  (e.g., isolation forest on training features) as a second-line
  filter is the standard mitigation.

### Out of scope, and why

The rubric's tool-misuse category is canonically about **agentic
systems calling external APIs** — booking flights, sending emails,
executing transactions. The classifier executes no external tools;
its only effect is to return a JSON response.

### How this category would apply if extended

If the service were extended into an agent that, say, automatically
flagged samples for human review or routed predictions to downstream
systems, the agentic tool-misuse risks would attach:

- *Excessive privilege*: the agent's bound credentials being used for
  actions outside its stated scope.
- *Confused-deputy attacks*: the agent acting on behalf of a malicious
  caller who manipulated its inputs.
- *Loop-of-action*: the agent repeatedly retrying a failed downstream
  call (or worse, a failing one that has side effects).

Each maps to standard mitigations from the agent-security literature
(least-privilege scoped tokens, principal-binding, idempotency keys,
circuit breakers).

## 5. Compliance concerns

### In scope

The C3 risk register documents the compliance risks that apply to
this system specifically (entries C-01 through C-05). Highlights:

- **C-01: Lack of model documentation prevents audit.**
  Mitigated by the model card (`docs/model-card.md`), the lineage
  diagram, and the audit trail.
- **C-02: Untracked model promotions.** Mitigated by the audit-trail
  schema requiring a `model_promoted_to_production` event for every
  artifact change, with the active version verifiable against the
  `iris_model_info` Prometheus gauge.
- **C-03: Insufficient power on A/B tests.** Mitigated by the C2
  experiment specification's explicit power calculation (5,715
  samples per arm for 1.0% MDE at α=0.05, power=0.80).
- **C-04: Monitoring data retention.** Mitigated by Prometheus's
  `--storage.tsdb.retention.time=2d` setting in
  `dashboards/monitoring-stack.yml`.
- **C-05: Right to explanation.** RandomForest is locally
  explainable; the model card recommends adding an explanation
  endpoint as a follow-up.

### Out of scope, and why

PII-specific regulation (GDPR, CCPA, HIPAA) does not bind this system
because no personal data is processed. The audit-trail design
(append-only, evidence-pointer schema) is intentionally compatible
with regulatory audit patterns so that *if* the patterns were reused
on a regulated system, the operational scaffolding wouldn't need to
be rebuilt.

### How this category would apply if extended

If the system handled regulated data, the existing audit-trail
schema would extend with: data-subject-access-request events,
deletion events with timestamps and verification, retention-policy
events tied to applicable regulation, and explicit consent-state
tracking per principal. The existing C3 framework (model card +
risk register + lineage + audit trail) is the right shape to
support those extensions.

## Summary

| Rubric category | Status | Where addressed |
|---|---|---|
| Data security | Partially in scope | Section 1 above + C3 risk register R-03, R-06; P-02 |
| Retrieval risks | Out of scope (no retriever) | Section 2 — explicit rationale + extension notes |
| Hallucination risks | In scope as "confidently wrong" classification | Section 3 + C4 drift report |
| Tool-misuse pathways | Partially in scope (the classifier as a tool) | Section 4 + C3 risk register R-02 |
| Compliance | In scope (model governance, not PII regulation) | Section 5 + C3 risk register C-01 through C-05 |

The risk matrix in `docs/risk-matrix.md` captures all in-scope risks
from this review with likelihood × severity ratings and concrete
mitigations. The CTO memo in `docs/cto-memo.md` summarizes findings
and action items for an executive audience.
