# SENTINEL — Industrial AI Platform

[![CI](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml)

SENTINEL is a predictive maintenance platform for industrial turbofan engines built on the NASA C-MAPSS benchmark dataset. It detects anomalies in sensor readings, estimates how many operational cycles remain before failure (Remaining Useful Life), and explains its predictions using SHAP and LIME — giving maintenance engineers both an early warning and a reason for it.

The project is structured in three layers:

- **ML engine** — a config-driven Python pipeline that trains anomaly detectors, RUL regressors, and explainability models across four C-MAPSS engine subsets
- **API backend** — a FastAPI server with JWT authentication, async database, real-time alerts via Redis, and Celery workers for background inference
- **Frontend dashboard** — a Next.js interface showing fleet health, per-asset RUL trends, SHAP attribution charts, and live alert management

Built as a PhD application portfolio for the FLARE project at University College Cork (Dr Ken Bruton, IERG).

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

## What the project does

**Anomaly detection** — two classical models (Isolation Forest, One-Class SVM) and an LSTM autoencoder trained on healthy sensor windows. The autoencoder learns what normal looks like, then flags sequences that deviate beyond a learned threshold.

**RUL estimation** — XGBoost regression trained on all 21 sensor channels plus operating settings. Predicts how many cycles remain before the engine fails. Enhanced with isotonic regression post-processing to enforce the physical constraint that RUL can only decrease over time.

**Explainability** — SHAP values show which sensors drove each RUL prediction. LIME provides an independent cross-check. Per-sensor reconstruction error from the LSTM autoencoder localises which channels degraded first.

**Operating Condition Normalisation** — the key research finding. Multi-condition subsets (FD002, FD004) have engines running at six different altitude/throttle regimes. Without normalising per regime, the autoencoder detects regime changes rather than degradation, collapsing F1 to near zero. KMeans clustering on operating settings followed by per-regime baseline normalisation fixes this.

**Health Index** — a scalar 0–1 score derived from PCA on healthy-phase sensor data. Smoother than raw RUL and useful for trending.

**Digital Twin** — a surrogate simulation model that lets you run what-if scenarios: what happens to RUL if this engine operates under a degraded profile for 50 more cycles?

**Maintenance metrics** — mean time to failure, early warning lead time, OEE impact, and maintenance ROI — framing the ML results in business terms.

**Dashboard** — fleet overview with health bars and live updates, asset detail with inference trigger, SHAP bar chart, RUL trend chart, and alert management with acknowledge/resolve workflow.

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

---

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

---

### 2. Download the dataset

```bash
# Download all four C-MAPSS subsets (FD001–FD004) from HuggingFace
python scripts/download_cmapss.py --subset all
```

This downloads to `data/` which is gitignored. Takes about a minute.

---

### 3. Run the ML pipeline

```bash
# Run a single subset
python -m ml.main --config configs/fd001.yaml

# Outputs go to:
#   reports/phase1/  — anomaly detection metrics and plots
#   reports/phase2/  — LSTM autoencoder metrics and plots
#   reports/phase3/  — SHAP charts, LIME comparison, reconstruction error

# Run all four subsets and reproduce the benchmark table
python scripts/run_benchmarks.py --auto-download

# Run with enhanced configs (OCNorm + monotonic RUL sweep for FD002/FD004)
python scripts/run_benchmarks.py --auto-download --tune-phase2
```

---

### 4. Start the API

Open a terminal, make sure the virtual environment is active:

```bash
pip install -r api/requirements.txt

uvicorn api.main:app --reload --port 8000
```

Seed demo data (run once):

```bash
python -m api.core.seed
```

The API is now running at **http://localhost:8000**

Interactive docs (try every endpoint): **http://localhost:8000/docs**

Demo credentials seeded:
| Email | Password | Role |
|-------|----------|------|
| admin@acme.com | Admin1234! | Admin |
| engineer@acme.com | Engineer1234! | Engineer |
| viewer@acme.com | Viewer1234! | Viewer |

---

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** and log in with `admin@acme.com` / `Admin1234!`

---

### 6. What you can do in the dashboard

| Page | What to try |
|------|-------------|
| `/dashboard` | See all 4 engines with health bars, RUL, and status. Refreshes live when inference runs. |
| `/assets` | Table view of the fleet — click any asset to go to its detail page. |
| `/assets/[id]` | Click **Score Now** to run ML inference on that asset. See anomaly score, RUL prediction, health index, and a SHAP bar chart showing which sensors drove the result. |
| `/alerts` | Switch between Open / Acknowledged / Resolved. Click the eye icon to acknowledge, the tick to resolve. |

---

### 7. Run the tests

```bash
# From the project root with the virtual environment active
python -m pytest tests/ -v
```

Expected: **13 passed**

---

### 8. Generate the evidence PDF

```bash
python scripts/generate_project_evidence.py
# Output: reports/application/project_evidence_ucc.pdf
```

---

### Docker (alternative — runs everything together)

Requires Docker Desktop:

```bash
# Production stack (Postgres + Redis + API + Celery worker)
docker compose up --build

# Development stack (SQLite + hot reload, no Celery)
docker compose -f docker-compose.dev.yml up
```

API docs: **http://localhost:8000/docs**

---

## Key Findings

**1. Operating Condition Normalisation is essential for multi-condition datasets.**
Without regime-aware normalisation, LSTM autoencoders detect regime shifts rather than degradation, collapsing anomaly F1 to near zero on FD002/FD004. KMeans clustering (k=6) on operating settings followed by per-regime normalisation resolves this entirely.

**2. Monotonic RUL enforcement reduces RMSE without retraining.**
Isotonic regression applied per-engine post-processing enforces the physical constraint that RUL is non-increasing. This yields −8.4% RMSE on FD002 and −12.8% on FD004 with no additional training cost.

**3. SHAP and LIME agree on top-5 features in >80% of test instances.**
Cross-method agreement validates explanation fidelity. Sensors 11, 12, 7, and 15 dominate attributions across all subsets — consistent with known HPC degradation signatures in C-MAPSS.

See [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) for the full write-up.

---

## Package Layout

```
ml/                    # AI engine (pure Python)
├── main.py            # Pipeline entry point
├── engine.py          # SentinelEngine — production inference interface
├── data/              # load, preprocess (OCNorm), windowing
├── models/
│   ├── anomaly/       # iforest, ocsvm, lstm_autoencoder
│   ├── rul/           # xgb_regressor, lstm_regressor
│   └── registry.py    # versioned model storage
├── explain/           # shap_explain, lime_explain, recon_error
├── simulation/        # digital_twin
├── eval/              # metrics (NASA score, maintenance KPIs), plots
└── utils/             # config validator, seed

api/                   # FastAPI backend
├── main.py            # App factory
├── core/              # config, database, security, redis, seed
├── models/            # SQLAlchemy ORM (8 tables)
├── schemas/           # Pydantic v2 request/response models
├── routers/           # auth, assets, sensors, inference, alerts,
│                      # analytics, work_orders, models, audit, websocket
├── services/          # inference_service, websocket_service
└── workers/           # Celery tasks (inference, training)

frontend/              # Next.js 16 + Tailwind CSS
├── app/               # login, dashboard, assets/[id], alerts
├── components/        # Nav, AuthGuard, charts, UI primitives
├── lib/               # typed API client, auth context
└── hooks/             # org-scoped WebSocket hook

configs/               # YAML pipeline configs (fd001–fd004, deploy/)
scripts/               # dataset download, benchmarking, deployment, evidence
tests/                 # API integration tests + ML unit tests
```

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is a turbofan engine degradation benchmark released by NASA. It simulates engines running from healthy to failure, recording 21 sensor channels (temperature, pressure, fan speed, fuel flow) and 3 operating settings per cycle. One cycle represents one flight.

There are four subsets:

| Subset | Engines (train) | Operating conditions | Fault mode |
|--------|:---------------:|:--------------------:|------------|
| FD001  | 100             | 1 (fixed)            | HPC degradation |
| FD002  | 260             | 6 (mixed)            | HPC degradation |
| FD003  | 100             | 1 (fixed)            | HPC + fan degradation |
| FD004  | 249             | 6 (mixed)            | HPC + fan degradation |

FD001 and FD003 are single-condition — every engine runs at the same altitude and throttle. FD002 and FD004 switch between six operating regimes mid-flight, making them significantly harder for sequence models that assume a consistent healthy baseline. This is the root cause of the near-zero anomaly F1 scores on those subsets before Operating Condition Normalisation was applied.

The dataset is downloaded automatically from HuggingFace via `scripts/download_cmapss.py`. `data/` is gitignored.

---

## Conclusion

The core finding of this project is that **operating condition awareness is the single most important factor** in making sequence-based anomaly detection work on real industrial datasets. Without it, LSTM autoencoders trained on multi-condition engines spend their capacity modelling regime changes rather than degradation — producing F1 scores close to zero regardless of model size or tuning. A lightweight pre-processing step (KMeans regime clustering + per-regime normalisation) resolves this entirely, improving FD002 anomaly F1 by over 900%.

The secondary finding is that **domain knowledge applied in post-processing** is often more cost-effective than architectural changes. Enforcing the physical constraint that RUL cannot increase reduced RMSE by up to 12.8% on the hardest subsets with no additional training.

Together these demonstrate that predictive maintenance models fail not because the ML is wrong but because the data preparation ignores the physics of the system — and that fixing the physics is faster than tuning the model.

---

## CI

[![CI](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml)

Python lint (ruff) + pytest + Next.js type-check + build on every push to `main`.
