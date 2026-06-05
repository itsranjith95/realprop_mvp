import os
import logging
from contextlib import contextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
DEFAULT_EXPERIMENT = "doc_type_classifier_mvp"
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "1") == "1"


def mlflow_available() -> bool:
    if not MLFLOW_ENABLED:
        return False

    if MLFLOW_TRACKING_URI.startswith("http://") or MLFLOW_TRACKING_URI.startswith("https://"):
        try:
            import requests
            r = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    return True


@contextmanager
def safe_mlflow_run(run_name: str, experiment_name: str = DEFAULT_EXPERIMENT, tags: Optional[dict] = None):
    if not mlflow_available():
        logger.warning("MLflow unavailable, running offline.")
        yield _NoOpRun()
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as e:
        logger.warning(f"MLflow unavailable ({e}), running offline.")
        yield _NoOpRun()


class _NoOpRun:
    class info:
        run_id = "offline-run"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def log_classification_params(vectorizer, classifier):
    if not mlflow_available():
        return
    try:
        import mlflow
        params = {}
        if hasattr(vectorizer, "get_params"):
            for k, v in vectorizer.get_params().items():
                params[f"tfidf_{k}"] = v
        if hasattr(classifier, "get_params"):
            for k, v in classifier.get_params().items():
                params[f"clf_{k}"] = v
        mlflow.log_params(params)
    except Exception as e:
        logger.debug(f"Could not log params: {e}")


def log_metrics_dict(metrics: dict):
    if not mlflow_available():
        return
    try:
        import mlflow
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
    except Exception as e:
        logger.debug(f"Could not log metrics: {e}")