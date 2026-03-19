# SENTINEL — Industrial AI Platform

[![CI](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml)

SENTINEL is a full-stack industrial AI platform — ML engine, REST API, and operator dashboard — that gives maintenance engineers both an early warning and a reason for it.

It detects anomalies in turbofan engine sensor readings, estimates how many operational cycles remain before failure (Remaining Useful Life), and explains every prediction using SHAP and LIME. Built on the NASA C-MAPSS benchmark dataset.

- **ML engine** — config-driven Python pipeline: anomaly detectors, RUL regressors, and explainability models across four C-MAPSS engine subsets
- **API backend** — FastAPI with JWT authentication, async database, real-time alerts via Redis, and Celery workers
- **Frontend dashboard** — Next.js interface with fleet health, per-asset RUL trends, SHAP attribution charts, and live alert management

---

## Results

### Anomaly Detection (F1 Score)

| Subset | Phase 1 iForest | Phase 2 LSTM AE | Phase 2 Enhanced | Key enhancement |
|--------|:---------:|:---------:|:-----------:|-----------------|
| FD001  | 0.5115    | 0.7408    | 0.7408      | —               |
| FD002  | 0.4487    | 0.0642    | **0.6126**  | OC Normalisation |
| FD003  | 0.4782    | 0.5434    | 0.5434      | —               |
| FD004  | 0.4537    | 0.0162    | **0.4519**  | OC Normalisation |

FD002 LSTM AE: **0.06 → 0.61** (+916%). FD004: **0.02 → 0.45** (+2,150%).

### RUL Estimation (RMSE — lower is better)

| Subset | Baseline | Enhanced | Improvement |
|--------|:--------:|:--------:|:-----------:|
| FD001  | 18.42    | 18.42    | —           |
| FD002  | 19.23    | **17.62** | −8.4%      |
| FD003  | 15.55    | 15.55    | —           |
| FD004  | 18.29    | **15.95** | −12.8%     |

All results are fully reproducible — see [Running the benchmarks](#3-run-the-ml-pipeline) below.

---

## Key Findings

**1. Operating Condition Normalisation is essential for multi-condition datasets.**
Without regime-aware normalisation, LSTM autoencoders detect regime shifts rather than degradation, collapsing anomaly F1 to near zero on FD002/FD004. KMeans clustering (k=6) on operating settings followed by per-regime normalisation resolves this entirely.

**2. Monotonic RUL enforcement reduces RMSE without retraining.**
Isotonic regression applied per-engine in post-processing enforces the physical constraint that RUL is non-increasing. This yields −8.4% RMSE on FD002 and −12.8% on FD004 with no additional training cost.

**3. SHAP and LIME agree on top-5 features in >80% of test instances.**
Cross-method agreement validates explanation fidelity. Sensors 11, 12, 7, and 15 dominate attributions across all subsets — consistent with known HPC degradation signatures in C-MAPSS.

See [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) for the full write-up.

---

## Architecture

```
┌──────────────────┬──────────────────┬───────────────────────────┐
│   ML Engine      │   FastAPI Backend │   Next.js Frontend        │
│   ml/            │   api/            │   frontend/               │
│                  │                  │                           │
│  Phase 1         │  Auth (JWT)       │  Fleet dashboard          │
│  ├ iForest       │  Assets CRUD      │  Asset detail + trends    │
│  ├ OCSVM         │  Sensor ingest    │  SHAP visualisation       │
│  └ XGBoost RUL   │  Inference API    │  Alert management         │
│                  │  Alerts           │  Real-time WebSocket      │
│  Phase 2         │  Work orders      │                           │
│  └ LSTM/GRU AE   │  Analytics        │                           │
│                  │  WebSocket        │                           │
│  Phase 3         │  Celery workers   │                           │
│  ├ SHAP          │  Redis pub/sub    │                           │
│  ├ LIME          │                  │                           │
│  └ Recon error   │                  │                           │
│                  │                  │                           │
│  Enhancements    │  SQLAlchemy async │  Framer Motion            │
│  ├ OCNorm        │  Postgres/SQLite  │  Recharts                 │
│  ├ Monotonic RUL │                  │  Tailwind CSS             │
│  ├ Health Index  │                  │                           │
│  └ Digital Twin  │                  │                           │
└──────────────────┴──────────────────┴───────────────────────────┘
```

---

## Running the project

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### 1. Clone and set up Python environment

```bash
git clone https://github.com/AhmadKhan46/sentinel-ai.git
cd sentinel-ai

python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python scripts/download_cmapss.py --subset all
```

Downloads to `data/` (gitignored). Takes about a minute.

### 3. Run the ML pipeline

```bash
# Single subset
python -m ml.main --config configs/fd001.yaml

# All four subsets — reproduce the benchmark table
python scripts/run_benchmarks.py --auto-download

# With OCNorm + monotonic RUL sweep (FD002/FD004 enhanced numbers)
python scripts/run_benchmarks.py --auto-download --tune-phase2
```

Outputs:
- `reports/phase1/` — anomaly detection metrics and plots
- `reports/phase2/` — LSTM autoencoder metrics and plots
- `reports/phase3/` — SHAP charts, LIME comparison, reconstruction error

### 4. Start the API

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000

# Seed demo data (run once)
python -m api.core.seed
```

API: **http://localhost:8000** · Docs: **http://localhost:8000/docs**

| Email | Password | Role |
|-------|----------|------|
| admin@acme.com | Admin1234! | Admin |
| engineer@acme.com | Engineer1234! | Engineer |
| viewer@acme.com | Viewer1234! | Viewer |

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** and log in with `admin@acme.com` / `Admin1234!`

| Page | What to try |
|------|-------------|
| `/dashboard` | Fleet overview with health bars and live inference updates |
| `/assets` | Table view — click any asset to open its detail page |
| `/assets/[id]` | Click **Score Now** to run inference. See anomaly score, RUL, health index, and SHAP chart |
| `/alerts` | Switch Open / Acknowledged / Resolved. Acknowledge and resolve with one click |

### 6. Run the tests

```bash
python -m pytest tests/ -v
```

Expected: **13 passed**

### Docker (alternative)

```bash
# Production stack (Postgres + Redis + API + Celery worker)
docker compose up --build

# Development stack (SQLite + hot reload)
docker compose -f docker-compose.dev.yml up
```

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) simulates turbofan engines running from healthy to failure, recording 21 sensor channels and 3 operating settings per cycle. One cycle = one flight.

| Subset | Engines (train) | Operating conditions | Fault mode |
|--------|:---------------:|:--------------------:|------------|
| FD001  | 100             | 1 (fixed)            | HPC degradation |
| FD002  | 260             | 6 (mixed)            | HPC degradation |
| FD003  | 100             | 1 (fixed)            | HPC + fan degradation |
| FD004  | 249             | 6 (mixed)            | HPC + fan degradation |

FD002 and FD004 switch between six operating regimes mid-flight — the root cause of near-zero anomaly F1 scores before Operating Condition Normalisation was applied.

Downloaded automatically via `scripts/download_cmapss.py`. `data/` is gitignored.

---

## Conclusion

The core finding is that **operating condition awareness is the single most important factor** in making sequence-based anomaly detection work on real industrial datasets. Without it, LSTM autoencoders spend their capacity modelling regime changes rather than degradation — producing F1 scores close to zero regardless of model size or tuning. A lightweight pre-processing step (KMeans regime clustering + per-regime normalisation) resolves this entirely, improving FD002 anomaly F1 by over 900%.

The secondary finding is that **domain knowledge applied in post-processing** is often more cost-effective than architectural changes. Enforcing the physical constraint that RUL cannot increase reduced RMSE by up to 12.8% on the hardest subsets with no additional training.

Predictive maintenance models fail not because the ML is wrong but because the data preparation ignores the physics of the system — and fixing the physics is faster than tuning the model.
