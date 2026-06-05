from pathlib import Path
import joblib

MODEL_PATH = Path("data/models/classification/tfidf_logreg.joblib")


class MLDocumentClassifier:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> dict:
        if self.model is None:
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

        try:
            predicted_label = self.model.predict([text])[0]

            all_scores = {}
            confidence = 0.0

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([text])[0]
                labels = list(self.model.classes_)
                all_scores = {
                    label: round(float(prob), 4)
                    for label, prob in zip(labels, probs)
                }
                confidence = round(float(max(probs)), 4)

            return {
                "doc_type": str(predicted_label),
                "confidence": confidence,
                "method": "ml_classifier",
                "error": None,
                "all_scores": all_scores,
            }

        except Exception as e:
            return {
                "doc_type": "other",
                "confidence": 0.0,
                "method": "ml_classifier",
                "error": str(e),
                "all_scores": {},
            }