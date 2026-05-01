# Risk Matrix — Component 5

## Scoring

- **Likelihood:** Low / Medium / High  
- **Severity:** Low / Medium / High  
- **Score:** 1–9 (used for prioritization)


## Key risks (in scope)

- **S-03 — Unauthenticated `/predict` endpoint**
  - Likelihood: High  
  - Severity: Medium  
  - Score: 6  
  - Mitigation: Add TLS, authentication, and rate limiting per user  

- **H-01 — “Confidently wrong” predictions under drift**
  - Likelihood: High  
  - Severity: Medium  
  - Score: 6  
  - Mitigation: Monitor PSI + add confidence vs. accuracy gap alert  

- **H-02 — Predictions on out-of-distribution (OOD) data without warning**
  - Likelihood: Medium  
  - Severity: Medium  
  - Score: 4  
  - Mitigation: Add OOD detector (e.g., isolation forest) and return “low confidence” instead of a normal prediction  

- **O-01 — Metrics go stale (dashboard looks healthy when it’s not)**
  - Likelihood: Medium  
  - Severity: Medium  
  - Score: 4  
  - Mitigation: Add `up == 0` alert in Prometheus to detect missing data  

- **O-03 — Drift baseline becomes outdated after retraining**
  - Likelihood: Medium  
  - Severity: Medium  
  - Score: 4  
  - Mitigation: Recompute and update baseline after each retrain  

- **S-01 — Pickle file swap → code execution risk**
  - Likelihood: Low  
  - Severity: High  
  - Score: 3  
  - Mitigation: Add SHA-256 hash verification at model load  

- **S-02 — Audit logs can be modified**
  - Likelihood: Low  
  - Severity: High  
  - Score: 3  
  - Mitigation: Move logs to append-only storage (e.g., S3 with object lock)  

- **C-02 — Model changes not tracked properly**
  - Likelihood: Low  
  - Severity: High  
  - Score: 3  
  - Mitigation: Enforce audit logging for every promotion  


## Lower-priority / acceptable risks

- **T-01 — Reverse-engineering via repeated queries**
  - Likelihood: Medium  
  - Severity: Low  
  - Mitigation: Add rate limiting if reused in a real system  

- **T-02 — Using model as a labeling oracle**
  - Likelihood: Medium  
  - Severity: Low  
  - Mitigation: Use scoped API access if needed  

- **O-02 — Cold-start latency spike**
  - Likelihood: Medium  
  - Severity: Low  
  - Mitigation: Send a warmup request at startup  

- **C-05 — No explanation endpoint**
  - Likelihood: Low  
  - Severity: Low  
  - Mitigation: Add SHAP/LIME endpoint if needed  



## Out-of-scope (by design)

These don’t apply because the system has no LLM, no retrieval, no external tools, and no PII:

- Retrieval risks (data leakage, stale data, contamination)  
- LLM hallucination (free-text generation)  
- External tool misuse (no API actions)  
- PII/regulatory risks  

If the system is extended (e.g., with RAG or real user data), these would need to be addressed.


## Summary

Top risks to focus on:

- Unauthenticated API access  
- Drift causing “confidently wrong” predictions  
- Missing OOD detection  
- Monitoring blind spots (stale metrics)  

All high-severity risks have clear, actionable mitigations that are realistic and would reduce risk.

The system is safe for its current scope, but to improve:

- Add API security (auth + rate limiting)  
- Improve drift/OOD detection  
- Strengthen audit logging and model integrity checks  

