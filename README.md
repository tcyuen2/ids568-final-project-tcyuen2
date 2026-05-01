# IDS568 Final Project — MLOps Capstone

This is my Final Project for IDS568. It builds a full operational framework around the Iris classifier I trained back in [Milestone 3](https://github.com/tcyuen2/ids568-milestone3-tcyuen2). The five components cover monitoring (C1), A/B testing (C2), governance (C3), drift detection (C4), and risk assessment (C5).

The point of the project isn't the model itself — Iris is a 150-sample teaching dataset and I'm using it as a stand-in. The point is the operational stuff around it: how do you monitor a model in production, how do you decide when to ship a new version, how do you document it for an audit, how do you catch drift before it breaks things, and how do you reason about risk at the system level.

## What's in this repo

- `src/monitoring/` — the FastAPI service that serves the model, plus all the Prometheus instrumentation
- `src/ab_test/` — the A/B test simulation and statistical analysis
- `src/drift/` — drift detection scripts that produce reference and production windows, then compute PSI / KS / accuracy impact
- `models/` — the bundled model artifacts pulled from the M3 MLflow run, plus the reference distribution stats used for drift PSI
- `dashboards/` — Docker Compose stack and Prometheus / Grafana config
- `docs/` — all the writeups (model card, memos, reports, diagrams)
- `logs/` — the audit trail
- `visualizations/` — dashboard screenshot and the C4 drift charts

## How to run it

You'll need Docker Desktop running and Python 3.11+ with a virtual environment.

First time setup:

```
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows; use source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

Spin up the full monitoring stack (this builds the FastAPI image and starts Prometheus and Grafana):

```
docker compose -f dashboards/monitoring-stack.yml up -d --build
```

Wait about 30 seconds for everything to come up healthy. Then in a separate terminal, generate some traffic:

```
python -m src.monitoring.traffic_gen --scenario normal --rate 5 --duration 300
```

The dashboard is at http://localhost:3000 (admin / admin). Prometheus is at http://localhost:9090. The FastAPI Swagger page is at http://localhost:8000/docs.

To see the drift panel react, change the scenario from `normal` to `drift` and run the traffic generator again. Within a couple minutes the petal-related features should cross the 0.25 PSI threshold.

To shut everything down:

```
docker compose -f dashboards/monitoring-stack.yml down
```

## Reproducing the analyses

The C2 A/B test is fully reproducible because everything is seeded:

```
python -m src.ab_test.simulation
python -m src.ab_test.analyze
```

That should produce `acc_A=0.9769`, `acc_B=0.9812`, decision `SHIP_B`. Same numbers every time as long as the seed stays at 42.

The C4 drift analysis is also reproducible:

```
python -m src.drift.generate_windows
python -m src.drift.detect
python -m src.drift.visualize
python -m src.drift.evidently_report
```

That produces 4 PNGs and 2 HTML reports in `visualizations/drift/`, plus the raw analysis output in `src/drift/results.json`.

## Where to find each component's writeup

C1 — `docs/dashboard-interpretation.md` (interpretation doc) and `dashboards/grafana-dashboards/iris-mlops-c1.json` (the dashboard itself, auto-loaded by Grafana). Screenshot in `visualizations/dashboard-overview.png`.

C2 — `docs/experiment-specification.md` (the design doc with the power calc) and `docs/recommendation-memo.md` (the ship/no-ship decision).

C3 — `docs/model-card.md`, `docs/risk-register.md`, `docs/lineage-diagram.png`, and `logs/audit-trail.json`.

C4 — `docs/drift-diagnostic-report.md` plus all the visualizations under `visualizations/drift/`.

C5 — `docs/system-boundary-diagram.png`, `docs/governance-review.md`, `docs/risk-matrix.md`, and `docs/cto-memo.md`.

## Lineage to Milestone 3

Everything in this repo traces back to a specific M3 MLflow run. The currently-active model is `baseline_run` from M3 run `0ab6a6fa80f949d5b917115218e126c4` (this got promoted as part of the C2 A/B test). The previous active model was `high_min_split` from run `6e01be84da0e47d68b1fc5caa9749e40`, which is still bundled in `models/variants/` for rollback.

The data SHA-256 hashes from M3's lineage report are referenced in the audit trail. The reference distribution used for drift PSI in C1 and C4 was computed from the same training set M3 used.

## Lessons learned

A few things I learned across the milestones that this project pulled together:

The biggest one is that **monitoring metrics need to come with interpretation**. I started C1 thinking the dashboard was the deliverable, but the dashboard panels alone don't tell anyone whether the system is healthy. The interpretation document is what makes the panels useful — connecting "the p99 is 10ms" to "this is fine for our SLA but leaves little margin." The same lesson hit me again in C4 when I realized that PSI numbers without an accuracy-impact analysis are basically meaningless to anyone trying to make a decision.

The second thing is that **small offline test sets lie**. The M3 test set was 30 samples and showed `high_min_split` (0.967) clearly beating `baseline_run` (0.933) — a 3.4-point gap. When I ran the C2 A/B test with ~5,700 samples per arm, the gap completely disappeared and `baseline_run` actually came out fractionally ahead. That's a useful real-world MLOps lesson: test sets in the dozens of samples have huge confidence intervals, and the gaps you see there often aren't real.

The third thing is the "confidently wrong" failure mode I found in C4. When I injected significant drift into the petal features, accuracy collapsed from 0.99 to 0.66, but mean prediction confidence stayed at 0.94. The model didn't *know* it was wrong. This is a known failure mode of tree-based models on out-of-distribution inputs, but seeing it happen in my own data made it click in a way that reading about it never did. It also explained why pure confidence-based monitoring isn't enough — you need input-distribution monitoring (which is what the PSI panel does) to catch these failures before they reach users.

A fourth thing is that **components should reference each other**. The strongest version of this project isn't five disconnected deliverables — it's a system where C1 monitors what C4 analyzes, C2 produces decisions that get logged in C3's audit trail, and C5 ties them all together. I tried to make those connections explicit in every document. It made the writing harder but the result is more cohesive.

Finally, **reproducibility is hard but worth it**. Pinned dependencies, seeded RNGs, scripts that regenerate diagrams from JSON — all of this took extra time but means the entire project can be rebuilt from scratch on a clean machine. That's a precondition for any real production deployment.

## Acknowledgments

Initial scaffolding for this project — the FastAPI monitoring service, drift detection scripts, A/B test simulation, governance documentation drafts, lineage and system-boundary diagrams — was developed with assistance from Claude (Anthropic). All design decisions, debugging, integration with the M3 system, dashboard construction, statistical interpretation, and the final editing of all written artifacts are my own work.
