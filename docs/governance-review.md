# Governance Review — Component 5

## 1. Data security

### In scope

- **API inputs:**  
  Currently unauthenticated → should add TLS, authentication, and rate limiting

- **Model files (`.pkl`):**  
  Pickle can execute code if tampered with which need hash validation at load

- **Audit trail:**  
  Stored as a normal file, it should move to append-only storage

- **Monitoring data:**  
  Safe here (no PII), but would need review in a real system

### Out of scope

- No PII so no GDPR/CCPA concerns

### If extended

Would require encryption, retention policies, and full data governance for user data.

---

## 2. Retrieval risks

### In scope

- None (no retrieval system)

### Why

No vector database, documents, or RAG pipeline, these risks don’t apply.

### If extended

Would need to handle:
- Data leakage across users  
- Cache poisoning  
- Stale data issues  

---

## 3. Hallucination risks

### In scope

- **“Confidently wrong” predictions**  
  Under drift, accuracy drops but confidence stays high

- Mitigation: drift monitoring (PSI) already detects this

### Out of scope

- No LLM → no text hallucination

### If extended

Would need grounding (RAG) and safeguards if adding LLM-based explanations.

---

## 4. Tool misuse

### In scope

Treating the model as a “tool,” risks include:

- Reverse-engineering via repeated queries  
- Using it as a labeling oracle  
- Adversarial inputs  

Mitigation: rate limiting, scoped access, possible OOD detection

### Out of scope

- No external tool execution

### If extended

Would require:
- Least-privilege access controls  
- Safeguards against unintended actions  
- Retry/circuit breaker mechanisms  

---

## 5. Compliance

### In scope

Covered through governance artifacts:

- Model card + lineage → auditability  
- Audit trail → tracks model changes  
- A/B testing → statistically sound decisions  
- Monitoring retention → configured  
- Explainability → possible with RandomForest  

### Out of scope

- No PII → no regulatory requirements

### If extended

Would need:
- Data access and deletion tracking  
- Consent management  
- Regulatory audit support  

---

## Summary

- **Data security:** relevant, needs improvements  
- **Retrieval risks:** not applicable  
- **Hallucination:** appears as “confidently wrong” predictions  
- **Tool misuse:** limited but possible via API abuse  
- **Compliance:** sufficient for current scope  

---

## Bottom line

The system is safe for its current use (non-sensitive data).

Recommended improvements:
- Secure model artifacts (hash checks)  
- Make audit logs append-only  
- Add rate limiting  

Additional safeguards would be needed for any regulated or higher-stakes deployment.