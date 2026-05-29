"""
Tests for Phase 3: Document Classification Pipeline
Run: pytest tests/test_classification.py -v
"""

import pytest
from src.classification import DocumentClassifier
from src.classification.rule_engine import RuleEngine
from src.classification.confidence import ConfidenceScorer


# ------------------------------------------------------------------
# Sample OCR texts (realistic snippets)
# ------------------------------------------------------------------

SALE_DEED_TEXT = """
This Sale Deed is executed on this 15th day of March 2023 between the vendor
Sri Ramesh Kumar and the vendee Sri Suresh Babu. The sale consideration is
Rs. 45,00,000 (Rupees Forty Five Lakhs only). The property is registered at
the Sub-Registrar office. Stamp duty has been paid as per the Karnataka
Stamp Act. The vendor hereby conveys the property absolutely.
"""

KHATA_CERT_TEXT = """
BBMP - Bruhat Bengaluru Mahanagara Palike
Khata Certificate
Khata No: 1234/56/78
This is to certify that the property tax assessment for the above khata number
has been duly registered in the name of Sri Anil Kumar.
Assessment Number: 2345678
Ward No: 123, Zone: South
"""

EC_TEXT = """
Sub-Registrar Office, Bangalore South
Encumbrance Certificate
Form 15
This certificate is issued to certify that there is no encumbrance
on the property for the period from 01-01-2015 to 31-12-2023.
Transactions registered: NIL
"""

AADHAR_TEXT = """
Government of India
Aadhaar
1234 5678 9012
Name: Priya Sharma
DOB: 15/08/1990
Female
VID: 9876543210123456
UIDAI
"""

EMPTY_TEXT = ""
GIBBERISH_TEXT = "asdf qwerty zxcv 12345 !@#$% random text with no meaning"


# ------------------------------------------------------------------
# Rule Engine Tests
# ------------------------------------------------------------------

class TestRuleEngine:

    def setup_method(self):
        self.engine = RuleEngine()

    def test_sale_deed_detected(self):
        result = self.engine.classify(SALE_DEED_TEXT)
        assert result["doc_type"] == "sale_deed"
        assert result["confidence"] > 0.3
        assert len(result["matched_keywords"]) > 0

    def test_khata_certificate_detected(self):
        result = self.engine.classify(KHATA_CERT_TEXT)
        assert result["doc_type"] == "khata_certificate"
        assert result["confidence"] > 0.3

    def test_ec_detected(self):
        result = self.engine.classify(EC_TEXT)
        assert result["doc_type"] == "encumbrance_certificate"

    def test_aadhar_detected(self):
        result = self.engine.classify(AADHAR_TEXT)
        assert result["doc_type"] == "aadhar_card"

    def test_unknown_for_gibberish(self):
        result = self.engine.classify(GIBBERISH_TEXT)
        assert result["doc_type"] == "unknown"

    def test_empty_text_returns_unknown(self):
        result = self.engine.classify("")
        assert result["doc_type"] == "unknown"
        assert result["confidence"] == 0.0


# ------------------------------------------------------------------
# Confidence Scorer Tests
# ------------------------------------------------------------------

class TestConfidenceScorer:

    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_agreement_bonus_applied(self):
        rule = {"doc_type": "sale_deed", "confidence": 0.6,
                "matched_keywords": [], "all_scores": {}}
        ollama = {"doc_type": "sale_deed", "confidence": 0.7,
                  "reasoning": "test", "method": "ollama", "model": "mistral", "error": None}
        result = self.scorer.merge(rule, ollama)
        # Both agree → should get agreement bonus
        assert result.doc_type == "sale_deed"
        assert result.confidence > 0.6

    def test_ollama_error_falls_back_to_rule(self):
        rule = {"doc_type": "khata_certificate", "confidence": 0.8,
                "matched_keywords": ["khata no"], "all_scores": {}}
        ollama = {"doc_type": "unknown", "confidence": 0.0,
                  "reasoning": "", "method": "ollama", "model": "mistral",
                  "error": "timeout"}
        result = self.scorer.merge(rule, ollama)
        assert result.doc_type == "khata_certificate"
        assert result.method == "rule_engine"

    def test_needs_human_review_flagged(self):
        rule = {"doc_type": "unknown", "confidence": 0.2,
                "matched_keywords": [], "all_scores": {}}
        result = self.scorer.merge(rule, None)
        assert result.needs_human_review is True

    def test_high_confidence_no_review(self):
        rule = {"doc_type": "sale_deed", "confidence": 0.9,
                "matched_keywords": ["sale deed"], "all_scores": {}}
        result = self.scorer.merge(rule, None)
        assert result.needs_human_review is False


# ------------------------------------------------------------------
# Full Classifier Integration (Rule Engine only — no Ollama needed)
# ------------------------------------------------------------------

class TestDocumentClassifier:

    def setup_method(self):
        # use_ollama=False so tests run without Ollama running
        self.clf = DocumentClassifier(use_ollama=False)

    def test_classify_sale_deed(self):
        result = self.clf.classify(SALE_DEED_TEXT)
        assert result.doc_type == "sale_deed"
        assert 0.0 <= result.confidence <= 1.0
        assert result.method == "rule_engine"

    def test_classify_empty_text(self):
        result = self.clf.classify(EMPTY_TEXT)
        assert result.doc_type == "unknown"
        assert result.error == "empty_text"
        assert result.needs_human_review is True

    def test_classify_returns_valid_doc_type(self):
        from src.classification.confidence import VALID_DOC_TYPES
        result = self.clf.classify(GIBBERISH_TEXT)
        assert result.doc_type in VALID_DOC_TYPES

    def test_batch_classify(self):
        docs = [
            {"doc_id": "doc1", "text": SALE_DEED_TEXT},
            {"doc_id": "doc2", "text": KHATA_CERT_TEXT},
            {"doc_id": "doc3", "text": EMPTY_TEXT},
        ]
        results = self.clf.classify_batch(docs)
        assert len(results) == 3
        assert results[0]["doc_type"] == "sale_deed"
        assert results[2]["doc_type"] == "unknown"

    def test_to_dict_serialisable(self):
        result = self.clf.classify(SALE_DEED_TEXT)
        d = result.to_dict()
        import json
        json.dumps(d)  # must not raise