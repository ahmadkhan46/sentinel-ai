.PHONY: dev api worker seed test lint train-fd001 train-all benchmark

# ── Development ───────────────────────────────────────────────────────────────
dev:
	docker compose -f docker-compose.dev.yml up

api:
	uvicorn api.main:app --reload --port 8000

worker:
	celery -A api.workers.celery_app worker --loglevel=info

seed:
	python -m api.core.seed

# ── ML ────────────────────────────────────────────────────────────────────────
train-fd001:
	python -m ml.main --config configs/fd001.yaml

train-fd002:
	python -m ml.main --config configs/fd002.yaml

train-fd003:
	python -m ml.main --config configs/fd003.yaml

train-fd004:
	python -m ml.main --config configs/fd004.yaml

train-all:
	python scripts/run_benchmarks.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-api:
	pytest tests/api/ -v

test-ml:
	pytest tests/ml/ -v

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	ruff check . && mypy api/ ml/

# ── Docker ────────────────────────────────────────────────────────────────────
up:
	docker compose up --build

down:
	docker compose down
