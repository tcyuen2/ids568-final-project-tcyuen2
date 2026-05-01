# Model Card — Iris Classifier

## Model overview

- **Model:** RandomForestClassifier (scikit-learn)  
- **Active version:** `baseline_run` (selected via A/B test)  
- **Classes:** setosa, versicolor, virginica  
- **Features:** 4 raw + 2 engineered (interaction terms)  


## Performance metrics

**Offline (small test set, n=30):**
- Accuracy: 0.933  
- Precision/Recall/F1 (macro): 0.933  
- AUC: 0.993  

**A/B test (simulated production, ~5,700 samples):**
- Accuracy: 0.9812  
- Compared to previous model: similar performance (slightly better)  
- Result: shipped (non-inferior)


## Training data

- **Source:** Classic Iris dataset (150 samples, balanced across 3 classes)  
- **Features:**
  - Sepal length, sepal width  
  - Petal length, petal width  
- **Engineered features:**
  - sepal_length × sepal_width  
  - petal_length × petal_width  
- **Preprocessing:** StandardScaler  
- **Data type:** Public, no PII  


## Intended use

- Demonstrate **MLOps workflows** (monitoring, A/B testing, drift detection)  
- Educational and experimentation purposes  

### Out of scope

- Real-world plant classification  
- Any high-stakes decisions (health, finance, etc.)  
- Production use with real user data  


## Limitations and failure modes

- Small dataset (150 samples): not representative of real-world data  
- Sensitive to drift: performance drops significantly if input distribution shifts  
- “Confidently wrong” behavior: model can output high confidence even when incorrect under drift  
- No automated retraining 


## Ethical risks and considerations

- **Low direct risk:** no personal or sensitive data involved  

If reused in real systems:
- **Bias risk:** would need subgroup evaluation  
- **Privacy risk:** monitoring metrics could expose data patterns  
- **Transparency:** should include model documentation and audit logs  


## Summary

This model is reliable within its controlled setup but mainly serves as a teaching example.  
Key risks are drift sensitivity and overconfidence under distribution shift.  
Additional safeguards would be needed for any real-world deployment.