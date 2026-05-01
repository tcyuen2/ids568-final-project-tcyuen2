"""
Prometheus metrics for the Iris classifier serving service.

All metric objects are defined here in one place. The serving app imports
them and updates them at appropriate points (request handlers, prediction
calls, drift recomputation hooks).

Design notes:
  - Histograms use buckets tuned to the *expected* range of values for
    this service. Generic buckets (Prometheus defaults) waste resolution
    where it's not needed and miss it where it is.
  - Labels are kept low-cardinality. High-cardinality labels (per-user,
    per-request-id) blow up memory in Prometheus and are an anti-pattern.
  - Metric names follow Prometheus conventions: lowercase, snake_case,
    base unit suffixes (_seconds, _total).
"""

from prometheus_client import Counter, Gauge, Histogram

# --------------------------------------------------------------------------- #
# RED metrics: Rate, Errors, Duration                                         #
# --------------------------------------------------------------------------- #

# Request count, broken down by endpoint and status. Allows error-rate
# computation as: errors / total over a time window.
REQUESTS_TOTAL = Counter(
    "iris_requests_total",
    "Total HTTP requests received by the prediction service.",
    labelnames=["endpoint", "status_code"],
)

# Total request latency (network + parsing + inference + serialization).
# Buckets chosen for a fast tabular model: most requests should sit
# in the 1-50ms range; anything past 500ms warrants investigation.
REQUEST_DURATION_SECONDS = Histogram(
    "iris_request_duration_seconds",
    "End-to-end HTTP request duration in seconds.",
    labelnames=["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Pure model inference time, isolated from the request lifecycle. If
# request duration is high but inference duration is low, the bottleneck
# is *not* the model — it's middleware, networking, or serialization.
# This separation is what turns "the API is slow" into a diagnosable problem.
INFERENCE_DURATION_SECONDS = Histogram(
    "iris_inference_duration_seconds",
    "Model inference duration in seconds (predict + predict_proba).",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5),
)

# In-flight gauge — captures concurrent load. Sustained high values
# combined with rising latency p99 indicates capacity pressure.
REQUESTS_IN_FLIGHT = Gauge(
    "iris_requests_in_flight",
    "Number of in-flight (concurrent) HTTP requests being processed.",
)

# --------------------------------------------------------------------------- #
# Input integrity signals                                                     #
# --------------------------------------------------------------------------- #

# Counter incremented whenever an input feature value is suspicious.
# 'anomaly_type' has fixed cardinality: nan, inf, out_of_range, missing.
# Spikes here typically precede prediction-quality degradation.
INPUT_ANOMALIES_TOTAL = Counter(
    "iris_input_anomalies_total",
    "Count of input integrity violations, by feature and type.",
    labelnames=["feature", "anomaly_type"],
)

# Live rolling-window mean of post-scaling feature values. After
# StandardScaler, training values were ~N(0,1), so a rolling mean
# drifting away from 0 is a strong, easy-to-read drift signal.
FEATURE_VALUE_MEAN = Gauge(
    "iris_feature_value_mean",
    "Rolling-window mean of post-scaling feature values.",
    labelnames=["feature"],
)

# Live rolling-window std of post-scaling feature values. Diverging
# from 1 indicates input variance is changing — could mean the input
# pipeline lost normalization, or the upstream data distribution shifted.
FEATURE_VALUE_STD = Gauge(
    "iris_feature_value_std",
    "Rolling-window standard deviation of post-scaling feature values.",
    labelnames=["feature"],
)

# --------------------------------------------------------------------------- #
# Output / drift signals                                                      #
# --------------------------------------------------------------------------- #

# Predicted class counter. Compare live ratios against training class
# proportions (1/3 each for stratified Iris). Sustained skew is an
# *output* drift indicator independent of feature drift.
PREDICTIONS_TOTAL = Counter(
    "iris_predictions_total",
    "Count of predictions emitted, by predicted class label.",
    labelnames=["predicted_class"],
)

# Histogram of max predicted probability (model confidence). Healthy
# distribution should be skewed toward 1.0 for an easy task like Iris.
# A leftward shift (more mass near 0.5) means the model is hedging more —
# a leading indicator of drift before accuracy actually drops.
PREDICTION_CONFIDENCE = Histogram(
    "iris_prediction_confidence",
    "Max-probability prediction confidence per request.",
    buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

# Per-feature PSI vs. training distribution. PSI thresholds (industry standard):
#   PSI < 0.1 : no significant drift
#   0.1 - 0.25: moderate drift, monitor
#   PSI > 0.25: significant drift, investigate / retrain
# Exposed as a gauge so dashboard alerts can fire on threshold crossings.
FEATURE_DRIFT_PSI = Gauge(
    "iris_feature_drift_psi",
    "Population Stability Index per feature vs. training distribution.",
    labelnames=["feature"],
)

# Output drift: PSI of live predicted class distribution vs. training.
# Single gauge (no labels) — the whole prediction distribution gets one number.
PREDICTION_DRIFT_PSI = Gauge(
    "iris_prediction_drift_psi",
    "PSI of live predicted-class distribution vs. training class proportions.",
)

# --------------------------------------------------------------------------- #
# Service info                                                                #
# --------------------------------------------------------------------------- #

# Metadata about the running service. Useful for dashboard headers and
# correlating metrics with specific model versions during incidents.
MODEL_INFO = Gauge(
    "iris_model_info",
    "Model metadata (always 1; labels carry the info).",
    labelnames=["model_version", "mlflow_run_id"],
)
