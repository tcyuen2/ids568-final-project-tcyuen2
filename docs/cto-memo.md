

## Executive summary

The Iris classifier is great working condition and fully operational. Monitoring, A/B testing, governance, and drift detenction are all online and workiing as intended.

There are no urgent bugs that need ficing, but there are a few improvements for our next steps.

## What was built

- Monitoring: Real-time visibility into latency, drift, and input quality
- A/B testing: Chose the better model using solid stats (not just offline results)
- Governance: Model card, audit trail, and lineage all in place
- Drift detection: Can catch distribution shifts and measure impact
- Risk review: No current red flags

## Key Improvement Areas

### 1. Model can be confidently wrong

Under distribution shift, the accuracy drops a lot but the model's confidence stays high.

### 2. Drifts alerts are too generic

Standard PSI thresholds don't fit all features equally. Some features matter way more than others

### 3. Audit trail isn't fully secure

Audit file is just a nortmal file that can be edited by anyone.

## Recommended actions to solve above problems

1. Add a confidence vs accuracy monitor to check the accuracy drops and monitor.
2. Tune thresholds per feature so drift alerts aren't too generic
3. Move audit to append-only storage to secure audit trail

## What this system does *not* tell us

- no real world risks, as we are inly using demo data
- no robustness testing
- no multi-tenant setup

