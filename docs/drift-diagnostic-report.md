# Drift Diagnostic Report — Component 4

1. Which features drifted most?

Drift is concentrated almost entirely in the petal-related features, while sepal features remain stable.

- Most drifted features:
   petal_length (largest shift)
   petal_width
   petal_l_x_petal_w (interaction term, amplified drift)
- Stable features:
   sepal_length
   sepal_width
   sepal_l_x_sepal_w

The interaction feature (petal_l_x_petal_w) shows significant drift because it compounds changes from both petal inputs. This highlights that engineered features can amplify underlying drift and should be monitored alongside raw features.

2. How does this impact model performance?

Drift in petal features has a strong negative impact on accuracy:

- No meaningful drift (baseline):
   Accuracy ≈ 0.98
- Moderate drift (1.2× scaling):
   Accuracy drops to ≈ 0.79 (~19% decrease)
- Severe drift (1.5× scaling + anomalies):
   Accuracy drops to ≈ 0.66 (~32% decrease)

A key issue is that the model becomes “confidently wrong” under drift:

- Prediction confidence remains high (~0.9+)
- Accuracy drops significantly

This means confidence alone is not a reliable signal of model health. Input-distribution monitoring (e.g., PSI) is necessary to detect these failures.

3. Recommended retraining or intervention

Immediate action
- Investigate upstream data sources, specifically those generating petal measurements. The isolated drift suggests a localized data issue (e.g., sensor or preprocessing change).

If drift is temporary
- Fix the upstream issue and continue using the current model.

If drift is persistent
- Retrain the model on updated data reflecting the new distribution
- Run an A/B test before deploying the new model
- Update the drift baseline after retraining to avoid false alerts
- Monitoring improvements
- Add per-feature drift thresholds (stricter for petal features)
- Add a confidence vs. accuracy gap monitor to detect “confidently wrong” behavior
- Strengthen input validation to handle anomalies (e.g., NaNs, extreme values)

Summary:

Drift is mostly affecting petal features and has a large impact on model accuracy. The current monitoring setup detects this correctly, but ther can be improvements in alerting and validation that would make the system more robust. The recommended approach is to first investigate the data and then retrain and recalibrate monitoring if the shift is permanent.