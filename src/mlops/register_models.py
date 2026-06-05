"""
Phase 8.1 — MLflow Model Registration
Registers trained classification model(s) into MLflow Model Registry
with experiment name 'doc_type_classifier_mvp' and appropriate tags.
Saves registry manifest to data/models/registry_manifest.json
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [register_models] %(message)s")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "doc_type_classifier_mvp"
REGISTERED_MODEL_NAME = "doc_type_classifier_mvp"

MODEL_DIR = Path("data/models/classification")
REGISTRY_MANIFEST = Path("data/models/registry_manifest.json")


def _read_eval_report(report_path: Path) -> dict:
    metrics = {}
    if not report_path.exists():
        return metrics

    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                metrics[key.strip()] = val.strip()
    return metrics


def _write_manifest(entries: list, status: str = "ok"):
    REGISTRY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "models": entries,
    }
    with open(REGISTRY_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Registry manifest → {REGISTRY_MANIFEST}")


def register_classification_model():
    model_path = MODEL_DIR / "tfidf_logreg.joblib"
    eval_report_path = MODEL_DIR / "eval_report.txt"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run classification_train stage first.")
        _write_manifest([], status="model_not_found")
        return

    try:
        import joblib
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        model = joblib.load(model_path)
        metrics = _read_eval_report(eval_report_path)

        run_id = None
        registered_version = None

        with mlflow.start_run(run_name="register_tfidf_logreg") as run:
            run_id = run.info.run_id

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            mlflow.set_tags({
                "project": "realprop_mvp",
                "phase": "8.1",
                "doc_type": "mother_deed_khata",
                "pipeline_stage": "classification",
                "framework": "sklearn",
                "model_type": "tfidf_logreg",
                "registered_at": datetime.now(timezone.utc).isoformat(),
            })

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                serialization_format="skops",
            )

            mlflow.log_artifact(str(model_path), artifact_path="saved_model")
            logger.info(f"Model registered — Run ID: {run_id}")

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            registered_version = latest.version
            logger.info(f"Registered model version: v{registered_version}")

            # Temporary: still valid today, but deprecated in MLflow.
            try:
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=registered_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(f"Model v{registered_version} transitioned to Production.")
            except Exception as stage_err:
                logger.warning(f"Stage transition skipped: {stage_err}")

        registry_entry = {
            "model_name": REGISTERED_MODEL_NAME,
            "experiment": EXPERIMENT_NAME,
            "run_id": run_id,
            "model_path": str(model_path),
            "metrics": metrics,
            "registered_version": registered_version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "status": "registered",
        }
        _write_manifest([registry_entry], status="ok")

    except Exception as e:
        logger.error(f"MLflow registration failed: {e}")
        logger.warning("Writing offline manifest.")
        offline_entry = {
            "model_name": REGISTERED_MODEL_NAME,
            "model_path": str(model_path),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "status": "offline_registration",
            "error": str(e),
        }
        _write_manifest([offline_entry], status="offline")


if __name__ == "__main__":
    register_classification_model()