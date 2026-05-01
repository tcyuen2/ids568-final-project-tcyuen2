# Dashboard Interpretation — Component 1

This document explains the production monitoring dashboard for the Iris classifier service. The goal is to answer: ho is the system doing overall? What problems might show up? What should trigger alerts?

## 1. Dashboard layout 

The dashboard is split into three layers to answwer differenty questions


System health- "Is the service up and fast?" | Request Rate, Request Latency p50/p95/p99 |

Capacity & internals - "If it's slow, where's the time going?" | Inference vs. Request Duration (p99) |

Data & model health - "Is what's coming in still what we trained on?" | Feature Drift PSI, Anomalies/sec, Prediction Class Distribution |




## 2. What the dashboard reveals about system health

Under healthy steady-state traffic at 5 req/s, the dashboard shows:

- **Latency**: p50 ~5 ms, p95 ~8 ms, p99 ~10 ms 
  The small gap between p95 and p99 means there are no real tail latency issues. The drift recompute lock isn't cauysing any contention at the load

- **Inference vs. request duration**: 
 the overlapping lines at p99 have less than 1ms difference. This tells us that the framework is tiny compared to model time.

- **Feature drift PSI**: 
all features are adound 0.05-0.2, which is below the significant drift threshold. Alerts should be based on our baseline and not 0.

- **Prediction distribution**: 
 Distribution is split almost evenly 3-ways (33%). When insertting anomalies, it becomes less equal because the extreme anomalies skew toward one class


## 3. Bottlenecks and risks

The dashboard has a few key risks:

**Tail Latency**- if request p99 increases but inference p99 stays flat then theres a problem in serving

**bad input data**- data quality risk. inserting anomalies caught a bug, where it wasnt counting anomalies correctly because they were NaN. Fixed by allow_nan=True


## 4. What would trigger an alert in production

- Feature drift PSI > 0.25 (10 min) → Page
  Significant drift, sustained (not just noise)
- Feature drift PSI > 0.1 (30 min) → Ticket
  Something’s changing, worth investigating
- Request p99 > 100 ms (5 min) → Page
  10× normal latency → real user impact
- 5xx errors > 1% → Page
  Service is failing in a meaningful way
- Input anomalies spike → Ticket
  Likely upstream data issue
- Prediction drift PSI > 0.25 (30 min) → Ticket
  Output distribution shifting unexpectedly

General idea:
- page when users are affected by latency or errors
- ticket when there are anomalies (or drift) but nothing is broken



