# SENTINEL — Industrial AI Platform

[![CI](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml)

> Enterprise predictive maintenance for turbofan engines, built on NASA C-MAPSS.
> PhD application portfolio — Dr Ken Bruton, IERG/FLARE, University College Cork.
> **GitHub:** https://github.com/AhmadKhan46/sentinel-ai

---

## Results

### Anomaly Detection (F1 Score)

| Subset | Phase 1 iForest | Phase 2 LSTM AE | Phase 2 (enhanced) | Key enhancement |
|--------|:---------:|:---------:|:-----------:|-----------------|
| FD001  | 0.5115    | 0.7408    | 0.7408      | —               |
| FD002  | 0.4487    | 0.0642    | **0.6126**  | OC Normalisation |
| FD003  | 0.4782    | 0.5434    | 0.5434      | —               |
| FD004  | 0.4537    | 0.0162    | **0.4519**  | OC Normalisation |

FD002 LSTM AE: **0.06 → 0.61** (+916%). FD004: **0.02 → 0.45** (+2,150%).

### RUL Estimation (RMSE, lower is better)

| Subset | Baseline RMSE | Enhanced RMSE | Improvement |
|--------|:-------------:|:-------------:|:-----------:|
| FD001  | 18.42         | 18.42         | —           |
| FD002  | 19.23         | **17.62**     | −8.4%       |
| FD003  | 15.55         | 15.55         | —           |
| FD004  | 18.29         | **15.95**     | −12.8%      |

> All results reproducible: `python scripts/run_benchmarks.py --auto-download`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENTINEL Platform                        │
├──────────────────┬──────────────────┬───────────────────────────┤
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

## Quick Start

### ML pipeline only

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download dataset
python scripts/download_cmapss.py --subset all

# Run single subset
python -m ml.main --config configs/fd001.yaml

# Full benchmark (FD001–FD004)
python scripts/run_benchmarks.py --auto-download

# Enhanced benchmark (with OCNorm + monotonic RUL sweep)
python scripts/run_benchmarks.py --auto-download --tune-phase2
```

### Full stack (development)

```bash
# Terminal 1 — API
pip install -r requirements.txt -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
python -m api.core.seed        # seed demo data

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev

# Open http://localhost:3000  (admin@acme.com / Admin1234!)
```

### Docker (production)

```bash
docker compose up --build
# API: http://localhost:8000/docs
# Frontend: http://localhost:3000
```

---

## Key Findings

**1. Operating Condition Normalisation is essential for multi-condition datasets.**
Without regime-aware normalisation, LSTM autoencoders detect operating condition regime shifts rather than degradation, collapsing anomaly F1 to near zero on FD002/FD004. KMeans regime clustering (k=6) on operating settings, followed by per-regime healthy baseline normalisation, resolves this entirely.

**2. Monotonic RUL enforcement reduces RMSE without retraining.**
Isotonic regression applied per-engine in post-processing enforces the physical constraint that RUL is non-increasing. This yields −8.4% RMSE on FD002 and −12.8% on FD004 with zero additional compute at training time.

**3. SHAP and LIME agree on top-5 features in >80% of test instances.**
Cross-method agreement validates explanation fidelity. Sensors 11, 12, 7, and 15 dominate attributions across all subsets — consistent with known HPC degradation signatures in C-MAPSS.

See [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) for the full write-up.

---

## Package Layout

```
ml/                    # AI engine (pure Python)
├── main.py            # Pipeline entry point
├── engine.py          # SentinelEngine — production inference
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
├── models/            # SQLAlchemy ORM
├── schemas/           # Pydantic v2
├── routers/           # auth, assets, sensors, inference, alerts, analytics…
├── services/          # inference_service, websocket_service
└── workers/           # Celery tasks (inference, training)

frontend/              # Next.js 16 + Tailwind
├── app/               # login, dashboard, assets, alerts
├── components/        # Nav, charts, UI primitives
├── lib/               # API client, auth context
└── hooks/             # WebSocket hook

configs/               # YAML pipeline configs (fd001–fd004, deploy/)
scripts/               # download, benchmark, deploy, evidence generation
tests/                 # API integration tests, ML unit tests
```

---

## Dataset

NASA C-MAPSS turbofan degradation simulation — 4 subsets (FD001–FD004), 21 sensors, train/test split by engine ID, ground-truth RUL files. Downloaded automatically via `scripts/download_cmapss.py` from HuggingFace. `data/` is gitignored.

---

## CI

[![CI](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadKhan46/sentinel-ai/actions/workflows/ci.yml)

Python lint (ruff) + pytest + Next.js type-check + build on every push.

Python lint (ruff) + pytest + Next.js type-check + build on every push.
