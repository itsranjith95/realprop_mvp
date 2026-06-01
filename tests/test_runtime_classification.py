from pathlib import Path

from src.classification.ml_classifier import MLDocumentClassifier


def test_model_file_exists_after_training():
    assert Path("models/classification/tfidf_logreg.joblib").exists()


def test_ml_classifier_predict_returns_shape():
    clf = MLDocumentClassifier()
    result = clf.predict("Khata certificate issued by BBMP with PID and ward details")
    assert "doc_type" in result
    assert "confidence" in result
    assert "method" in result