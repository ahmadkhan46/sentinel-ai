# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based ML research application for fault detection, diagnostics, and RUL (Remaining Useful Life) estimation using NASA's C-MAPSS turbofan engine dataset. Designed as a PhD application portfolio project in predictive maintenance and explainable AI.

## Common Commands

**Setup:**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Download dataset:**
```bash
python scripts/download_cmapss.py --subset FD001   # single subset
python scripts/download_cmapss.py --subset all     # all subsets (FD001–FD004)
```

**Run pipeline:**
```bash
python -m ml.main --config configs/fd001.yaml
```

**Benchmarking:**
```bash
python scripts/run_benchmarks.py --auto-download                  # FD001–FD004 baseline
python scripts/run_benchmarks.py --auto-download --tune-phase2    # include Phase 2 config sweep
```

**Generate deployment configs and reports:**
```bash
python scripts/generate_deploy_configs.py   # create per-subset deployment configs
python scripts/run_deploy_manifest.py       # execute all deployment configs
python scripts/generate_project_evidence.py # generate one-page application evidence brief
```

**API backend (FastAPI):**
```bash
uvicorn api.main:app --reload --port 8000   # dev server
python -m api.core.seed                     # seed demo data (Acme org, 3 users, 4 assets)
pytest tests/ -v                            # run API + ML tests
```

**Frontend (Next.js):**
```bash
cd frontend && npm run dev    # dev server on :3000
cd frontend && npm run build  # production build check
```

**Docker (full stack):**
```bash
docker compose up --build         # Postgres + Redis + API + Celery worker
docker compose -f docker-compose.dev.yml up  # SQLite + hot-reload
```

There is no ML test suite separate from benchmarks — ML correctness is validated through benchmark runs and config validation. API tests use pytest + httpx with an in-memory SQLite database.

## Package Layout

```
ml/                              # AI engine — pure Python, no web framework
├── main.py                      # Pipeline entry point (python -m ml.main --config ...)
├── engine.py                    # SentinelEngine — production inference interface
├── data/                        # load_cmapss, preprocess, windowing
├── models/
│   ├── anomaly/                 # iforest, ocsvm, lstm_autoencoder
│   ├── rul/                     # xgb_regressor, lstm_regressor
│   └── registry.py              # ModelRegistry (save/load/version)
├── explain/                     # shap_explain, lime_explain, recon_error
├── simulation/                  # digital_twin
├── eval/                        # metrics, plots
└── utils/                       # config (YAML validator), seed
```

The `src/` directory is the old location and is superseded by `ml/`. Do not add new code there.

```
api/                             # FastAPI backend
├── main.py                      # App factory (create_app), lifespan, CORS, router registration
├── core/                        # config (Settings), database (async SQLAlchemy), security (JWT/bcrypt), redis, seed
├── models/                      # SQLAlchemy ORM: organisation, user, asset, sensor_reading, inference_result, alert, work_order, audit_log
├── schemas/                     # Pydantic v2 schemas (request/response)
├── routers/                     # auth, organisations, assets, sensors, inference, alerts, analytics, work_orders, models, audit, ws
├── services/                    # inference_service (score + alert), websocket_service (ConnectionManager)
├── middleware/                  # auth (JWT bearer), audit (log_action)
└── workers/                     # celery_app, inference_tasks, training_tasks

frontend/                        # Next.js 16 + Tailwind frontend
├── app/                         # App Router pages: /, /login, /dashboard, /assets, /assets/[id], /alerts
├── components/                  # Nav, AuthGuard, ui/ (Badge, Card, StatCard, Spinner), charts/ (TrendChart, ShapBar)
├── lib/                         # api.ts (typed fetch client), auth.tsx (AuthProvider + useAuth)
└── hooks/                       # useWebSocket (org-scoped real-time inference events)
```

## Architecture

The pipeline runs in 3 phases, all orchestrated via `ml/main.py`:

**Phase 1 — Baseline models:**
- Data loaded from `ml/data/` (load → preprocess → split by engine ID)
- Anomaly detection: Isolation Forest + One-Class SVM with proxy labels (last N cycles = anomalous)
- RUL regression: XGBoost trained on sensor + operating-condition features
- Outputs metrics to `reports/phase1/`

**Phase 2 — Sequence anomaly detection:**
- `ml/data/windowing.py` creates overlapping 3D time-window tensors
- LSTM or GRU autoencoder (`ml/models/anomaly/lstm_autoencoder.py`) trained on healthy windows only
- Anomaly scored by reconstruction error vs. quantile threshold
- `run_benchmarks.py` sweeps ~13 config profiles (model type, threshold, window size, healthy fraction, anomaly horizon) and auto-selects best per subset
- Outputs to `reports/phase2/`

**Phase 3 — Explainability:**
- SHAP feature attribution for the XGBoost RUL model (`ml/explain/shap_explain.py`)
- Per-sensor reconstruction error diagnostics for anomalous sequences (`ml/explain/recon_error.py`)
- Outputs to `reports/phase3/`

## Configuration

All runs are controlled by YAML configs (`configs/fd001.yaml`, `configs/deploy/*.yaml`). `ml/utils/config.py` performs strict schema validation before execution — type errors and out-of-range values are caught and reported upfront.

Key config knobs:
- `dataset.sequence_length` — time window size (default 30 cycles)
- `dataset.target_rul_clip` — max RUL cap (default 125)
- `phase2.model_type` — `"lstm"` or `"gru"`
- `phase2.threshold_quantile` — anomaly score cutoff (default 0.97)
- `anomaly.iforest.contamination` / `anomaly.ocsvm.nu` — baseline model contamination
- `rul.xgboost.*` — tree depth, learning rate, subsampling

`configs/deploy/manifest.json` stores best-config recommendations produced by the benchmark sweep.

## Key Metrics

- **Anomaly:** Precision / Recall / F1 (binary labels from proxy labeling)
- **RUL:** MAE, RMSE, and NASA asymmetric score — penalizes late-maintenance predictions (overestimation) more heavily than early ones via `exp(-δ/13)−1` / `exp(δ/10)−1`

## Data

`data/` is gitignored. C-MAPSS subsets (FD001–FD004) are downloaded from HuggingFace via `scripts/download_cmapss.py`. Each subset has train/test text files plus a ground-truth RUL file. FD001 and FD003 are single-condition; FD002 and FD004 are multi-condition (harder).
