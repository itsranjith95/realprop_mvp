from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.classification.ml_classifier import MLDocumentClassifier

TEST_PATH = Path("data/labeled/classification/test.csv")
REPORT_PATH = Path("models/classification/eval_report.txt")


def main():
    df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])
    clf = MLDocumentClassifier()

    if not clf.is_available():
        raise RuntimeError("Model not available. Train first.")

    preds = [clf.predict(text)["doc_type"] for text in df["text"].astype(str)]
    y_true = df["label"].astype(str).tolist()

    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds)
    cm = confusion_matrix(y_true, preds)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        f"Accuracy: {acc:.4f}\n\n"
        f"Classification Report:\n{report}\n\n"
        f"Confusion Matrix:\n{cm}\n",
        encoding="utf-8",
    )

    print(f"Accuracy: {acc:.4f}")
    print(report)
    print(cm)


if __name__ == "__main__":
    main()