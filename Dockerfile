# --------------------------------------------------------------------------- #
# Iris classifier serving image.                                              #
#                                                                             #
# Stays small (~150MB) by:                                                    #
#   - Using python:3.11-slim base                                             #
#   - Installing only the serving-time deps (no pandas, statsmodels, etc.)    #
#   - Copying only src/monitoring + models (not docs / dashboards / tests)    #
# --------------------------------------------------------------------------- #

FROM python:3.11-slim

WORKDIR /app

# System deps: build tools needed only for sklearn wheels on some platforms.
# Install then clean apt cache to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Serving-time only requirements (subset of full requirements.txt).
# Pinning the exact versions used at training time ensures pickle compat.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy app code and bundled model artifacts. Note: we deliberately do not
# copy docs/, dashboards/, etc. so the runtime image stays minimal.
COPY src/monitoring/ ./src/monitoring/
COPY models/ ./models/

# Run as non-root for a small security improvement.
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker is fine for a teaching demo and keeps the in-process
# rolling window deterministic. For real production, multi-worker
# requires either pushing rolling state to a shared store (Redis) or
# accepting per-worker drift estimates that the Prometheus scrape
# aggregates anyway.
CMD ["uvicorn", "src.monitoring.app:app", "--host", "0.0.0.0", "--port", "8000"]
