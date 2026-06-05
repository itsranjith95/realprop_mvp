"""
Phase 8.1 — Classification Train Stage
Trains TF-IDF + LogisticRegression classifier on labeled data.
Logs params, metrics, and model to MLflow experiment 'doc_type_classifier_mvp'.
Saves model to data/models/classification/tfidf_logreg.joblib (DVC-tracked).
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from src.mlops.mlflow_utils import (
    safe_mlflow_run,
    log_classification_params,
    log_metrics_dict,
    mlflow_available,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [train_classifier] %(message)s")
logger = logging.getLogger(__name__)

TRAIN_CSV = Path("data/labeled/classification/train.csv")
VAL_CSV = Path("data/labeled/classification/val.csv")
MODEL_OUT = Path("data/models/classification/tfidf_logreg.joblib")

TEXT_CANDIDATES = ["text", "ocr_text", "content"]
LABEL_CANDIDATES = ["doc_type", "label", "document_type", "type", "class"]


def resolve_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {kind} column. Found columns={df.columns.tolist()}")


def load_and_resolve(path: Path):
    if not path.exists():
        logger.warning(f"{path} not found — using stub data.")
        df = pd.DataFrame({
            "text": ["mother deed property bengaluru", "khata certificate bbmp"],
            "label": ["mother_deed", "khata_certificate"],
        })
    else:
        df = pd.read_csv(path)

    text_col = resolve_column(df, TEXT_CANDIDATES, "text")
    label_col = resolve_column(df, LABEL_CANDIDATES, "label")
    return df, text_col, label_col


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def train():
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_df, train_text_col, train_label_col = load_and_resolve(TRAIN_CSV)
    val_df, val_text_col, val_label_col = load_and_resolve(VAL_CSV)

    X_train = train_df[train_text_col].fillna("").astype(str)
    y_train = train_df[train_label_col].fillna("other").astype(str)

    X_val = val_df[val_text_col].fillna("").astype(str)
    y_val = val_df[val_label_col].fillna("other").astype(str)

    pipeline = build_pipeline()

    tags = {
        "phase": "8.1",
        "stage": "classification_train",
        "model_type": "tfidf_logreg",
        "dataset": "bengaluru_property_docs",
    }

    with safe_mlflow_run("train_tfidf_logreg", tags=tags) as run:
        logger.info(f"MLflow run: {run.info.run_id}")
        logger.info(f"Train text column: {train_text_col} | label column: {train_label_col}")
        logger.info(f"Val text column: {val_text_col} | label column: {val_label_col}")

        log_classification_params(pipeline.named_steps["tfidf"], pipeline.named_steps["clf"])

        logger.info(f"Training on {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        metrics = {
            "val_accuracy": acc,
            "val_f1_weighted": f1,
        }
        log_metrics_dict(metrics)

        logger.info(f"Val accuracy: {acc:.4f}")
        logger.info(f"Val F1 (weighted): {f1:.4f}")

        joblib.dump(pipeline, MODEL_OUT)
        logger.info(f"Model saved → {MODEL_OUT}")

        if mlflow_available():
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name="model",
                    serialization_format="skops",
                )
            except Exception as e:
                logger.warning(f"Skipping MLflow model log: {e}")

    return pipeline


if __name__ == "__main__":
    train()