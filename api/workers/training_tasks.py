from __future__ import annotations

import subprocess
import sys

from api.workers.celery_app import celery


@celery.task(name="training.run_pipeline", bind=True, max_retries=1, time_limit=3600)
def run_training_pipeline(self, config: str = "configs/fd001.yaml") -> dict:
    """Trigger an ML training pipeline run as a subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "ml.main", "--config", config],
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],  # last 4k chars
        "stderr": result.stderr[-2000:],
    }
