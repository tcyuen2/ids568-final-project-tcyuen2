# IDS568 Final Project — MLOps Capstone

A production operations framework wrapped around the Iris classifier from
[Milestone 3](https://github.com/tcyuen2/ids568-milestone3-tcyuen2).

> Status: in progress. This README will be expanded as each component is completed.

## What's here so far

**Component 1 — Production Monitoring Dashboard**

- Instrumented FastAPI service serving the M3 RandomForest model
- Prometheus metrics: latency, errors, throughput, input integrity, per-feature drift PSI
- Docker Compose stack (Prometheus + Grafana + service)
- Synthetic traffic generator with normal / drift / anomaly scenarios

## Quick start

```bash
# 1. Spin up the full stack
docker compose -f dashboards/monitoring-stack.yml up -d --build

# 2. In a separate terminal, generate traffic
pip install -r requirements.txt
python -m src.monitoring.traffic_gen --scenario drift --rate 5 --duration 300

# 3. Open the dashboards
open http://localhost:3000  # Grafana (admin / admin)
open http://localhost:9090  # Prometheus
open http://localhost:8000/docs  # FastAPI Swagger
```

## Lineage

| Stage | Source |
|-------|--------|
| Model | M3 MLflow run `6e01be84da0e47d68b1fc5caa9749e40` |
| Training data | sklearn Iris + 2 interaction features |
| Scaler | StandardScaler from same M3 run |

## Layout

```
src/monitoring/   # Component 1: instrumented serving service
src/ab_test/      # Component 2: A/B simulation (TODO)
src/drift/        # Component 4: drift detection (TODO)
models/           # Bundled model + scaler + reference distribution
dashboards/       # Docker Compose + Prometheus + Grafana configs
docs/             # Component reports (TODO)
logs/             # Audit trail (TODO)
visualizations/   # Drift charts and dashboard screenshots (TODO)
```

(See the section per component once they're written.)
