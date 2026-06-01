from pathlib import Path
from typing import Optional

import joblib

MODEL_DIR = Path("models/classification")
MODEL_PATH = MODEL_DIR / "tfidf_logreg.joblib"


class MLDocumentClassifier:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model_path = Path(model_path)
        self.bundle = None

        if self.model_path.exists():
            self.bundle = joblib.load(self.model_path)

    def is_available(self) -> bool:
        return self.bundle is not None

    def predict(self, text: str) -> dict:
        if not self.bundle:
            return {
                "doc_type": "other",
                "confidence": 0.0,
                "method": "ml_classifier",
                "error": "model_not_loaded",
                "all_scores": {},
            }

        if not text or not text.strip():
            return {
                "doc_type": "other",
                "confidence": 0.0,
                "method": "ml_classifier",
                "error": "empty_text",
                "all_scores": {},
            }

        vectorizer = self.bundle["vectorizer"]
        model = self.bundle["model"]
        labels = self.bundle["labels"]

        X = vectorizer.transform([text])
        probs = model.predict_proba(X)[0]
        pred_idx = probs.argmax()
        pred_label = labels[pred_idx]
        pred_conf = float(probs[pred_idx])

        all_scores = {
            label: round(float(prob), 4)
            for label, prob in zip(labels, probs)
        }

        return {
            "doc_type": pred_label,
            "confidence": round(pred_conf, 4),
            "method": "ml_classifier",
            "error": None,
            "all_scores": all_scores,
        }