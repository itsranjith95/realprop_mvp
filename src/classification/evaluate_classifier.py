"""
Phase 8.1 — Classification Evaluate Stage
Evaluates tfidf_logreg.joblib on test set.
Logs per-class metrics to MLflow and writes eval_report.txt (DVC-tracked).
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.mlops.mlflow_utils import safe_mlflow_run, log_metrics_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [evaluate_classifier] %(message)s")
logger = logging.getLogger(__name__)

TEST_CSV = Path("data/labeled/classification/test.csv")
MODEL_PATH = Path("data/models/classification/tfidf_logreg.joblib")
EVAL_REPORT = Path("data/models/classification/eval_report.txt")

TEXT_CANDIDATES = ["text", "ocr_text", "content"]
LABEL_CANDIDATES = ["doc_type", "label", "document_type", "type", "class"]


def resolve_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {kind} column. Found columns={df.columns.tolist()}")


def load_and_resolve(path: Path):
    if not path.exists():
        logger.warning(f"{path} not found — using stub.")
        df = pd.DataFrame({
            "text": ["mother deed property bengaluru", "khata certificate bbmp"],
            "label": ["mother_deed", "khata"],
        })
    else:
        df = pd.read_csv(path)

    text_col = resolve_column(df, TEXT_CANDIDATES, "text")
    label_col = resolve_column(df, LABEL_CANDIDATES, "label")
    return df, text_col, label_col


def evaluate():
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}.")
        EVAL_REPORT.write_text("error: model not found\n")
        return

    pipeline = joblib.load(MODEL_PATH)
    test_df, text_col, label_col = load_and_resolve(TEST_CSV)

    X_test = test_df[text_col].fillna("")
    y_test = test_df[label_col].astype(str)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    logger.info(f"Test text column: {text_col} | label column: {label_col}")
    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info(f"Test F1 (weighted): {f1:.4f}")
    logger.info(f"\n{report}")

    report_text = (
        f"test_accuracy: {acc:.6f}\n"
        f"test_f1_weighted: {f1:.6f}\n"
        f"\nClassification Report:\n{report}"
    )
    EVAL_REPORT.write_text(report_text, encoding="utf-8")
    logger.info(f"Eval report → {EVAL_REPORT}")

    tags = {
        "phase": "8.1",
        "stage": "classification_evaluate",
    }
    with safe_mlflow_run("evaluate_tfidf_logreg", tags=tags):
        log_metrics_dict({"test_accuracy": acc, "test_f1_weighted": f1})

    return {"accuracy": acc, "f1_weighted": f1}


if __name__ == "__main__":
    evaluate()