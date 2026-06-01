from src.classification.ml_classifier import MLDocumentClassifier


def test_ml_classifier_loads_or_handles_missing_model():
    clf = MLDocumentClassifier()
    assert clf is not None


def test_ml_classifier_empty_text():
    clf = MLDocumentClassifier()
    result = clf.predict("")
    assert "doc_type" in result
    assert "confidence" in result
    
    