# tests/test_phase5.py
"""
Unit tests for Phase 5:
  - Cross-document Validation Pipeline
  - Rules & Risk Scoring Pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.core.validation_models import RiskLabel, Severity, ValidationType
from src.pipelines.rules_pipeline import evaluate_rules, run_rules_pipeline
from src.pipelines.validation_pipeline import (
    collect_validation_results,
    validate_area,
    validate_owner_names,
    validate_property_ids,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOOD_ENTITIES: Dict[str, Any] = {
    "mother_deed": {
        "buyer_name": "Ranjith Kumar",
        "seller_name": "Suresh Reddy",
        "property_id": "SY NO 45/2",
        "area": "1200 sqft",
        "extraction_confidence": {
            "buyer_name": 0.95,
            "seller_name": 0.92,
            "property_id": 0.88,
            "area": 0.90,
        },
    },
    "khata": {
        "owner_name": "Ranjith Kumar",
        "property_id": "SY NO 45/2",
        "area": "1200 sqft",
        "extraction_confidence": {
            "owner_name": 0.91,
            "property_id": 0.87,
            "area": 0.89,
        },
    },
}

MISMATCH_ENTITIES: Dict[str, Any] = {
    "mother_deed": {
        "buyer_name": "Ranjith Kumar",
        "seller_name": "Suresh Reddy",
        "property_id": "SY NO 45/2",
        "area": "1200 sqft",
    },
    "khata": {
        "owner_name": "Mahesh Babu",           # name mismatch
        "property_id": "SY NO 99/1",           # property_id mismatch
        "area": "900 sqft",                    # area mismatch
    },
}

MISSING_ENTITIES: Dict[str, Any] = {
    "mother_deed": {
        "buyer_name": None,
        "seller_name": None,
        "property_id": None,
        "area": None,
    },
    "khata": {
        "owner_name": None,
        "property_id": None,
        "area": None,
    },
}


# ---------------------------------------------------------------------------
# validate_owner_names
# ---------------------------------------------------------------------------

class TestValidateOwnerNames:
    def test_match(self):
        result = validate_owner_names(GOOD_ENTITIES)
        assert result.passed is True
        assert result.type == ValidationType.OWNER_NAME

    def test_mismatch(self):
        result = validate_owner_names(MISMATCH_ENTITIES)
        assert result.passed is False
        assert result.rule_id == "OWNER_NAME_MISMATCH"
        assert result.severity == Severity.HIGH

    def test_missing_both(self):
        result = validate_owner_names(MISSING_ENTITIES)
        assert result.passed is False
        assert result.rule_id == "MISSING_OWNER_NAME"
        assert result.severity == Severity.CRITICAL
        assert result.mandatory_review is True

    def test_fuzzy_partial_match(self):
        entities = {
            "mother_deed": {"buyer_name": "Ranjith Kumar S"},
            "khata": {"owner_name": "Ranjith Kumar"},
        }
        result = validate_owner_names(entities)
        assert result.passed is True  # minor suffix should still match

    def test_evidence_populated(self):
        result = validate_owner_names(GOOD_ENTITIES)
        assert "fuzzy_score" in result.evidence


# ---------------------------------------------------------------------------
# validate_property_ids
# ---------------------------------------------------------------------------

class TestValidatePropertyIds:
    def test_match(self):
        result = validate_property_ids(GOOD_ENTITIES)
        assert result.passed is True

    def test_mismatch(self):
        result = validate_property_ids(MISMATCH_ENTITIES)
        assert result.passed is False
        assert result.rule_id == "PROPERTY_ID_MISMATCH"

    def test_missing_both(self):
        result = validate_property_ids(MISSING_ENTITIES)
        assert result.passed is False
        assert result.rule_id == "MISSING_PROPERTY_ID"

    def test_normalisation(self):
        entities = {
            "mother_deed": {"property_id": "sy no 45 / 2"},
            "khata": {"property_id": "SYNO45/2"},
        }
        result = validate_property_ids(entities)
        # After normalization both become "SYNO45/2" or similar
        assert result.passed is True


# ---------------------------------------------------------------------------
# validate_area
# ---------------------------------------------------------------------------

class TestValidateArea:
    def test_match(self):
        result = validate_area(GOOD_ENTITIES)
        assert result.passed is True

    def test_mismatch(self):
        result = validate_area(MISMATCH_ENTITIES)
        assert result.passed is False
        assert result.rule_id == "AREA_MISMATCH"

    def test_missing_both(self):
        result = validate_area(MISSING_ENTITIES)
        # MISSING_AREA is non-blocking → passed=True
        assert result.passed is True
        assert result.rule_id == "MISSING_AREA"

    def test_within_tolerance(self):
        entities = {
            "mother_deed": {"area": "1200 sqft"},
            "khata": {"area": "1210 sqft"},   # 0.83% diff
        }
        result = validate_area(entities)
        assert result.passed is True

    def test_outside_tolerance(self):
        entities = {
            "mother_deed": {"area": "1200 sqft"},
            "khata": {"area": "1000 sqft"},   # 20% diff
        }
        result = validate_area(entities)
        assert result.passed is False


# ---------------------------------------------------------------------------
# collect_validation_results
# ---------------------------------------------------------------------------

class TestCollectValidationResults:
    def test_good_case_all_pass(self):
        results = collect_validation_results("CASE001", GOOD_ENTITIES)
        failures = [r for r in results if not r.passed]
        assert len(failures) == 0

    def test_mismatch_case_has_failures(self):
        results = collect_validation_results("CASE002", MISMATCH_ENTITIES)
        failures = [r for r in results if not r.passed]
        assert len(failures) >= 2   # owner name + property id + area

    def test_missing_case_max_failures(self):
        results = collect_validation_results("CASE003", MISSING_ENTITIES)
        failures = [r for r in results if not r.passed]
        assert len(failures) >= 3


# ---------------------------------------------------------------------------
# evaluate_rules
# ---------------------------------------------------------------------------

class TestEvaluateRules:
    def test_green_on_good_case(self):
        validations = collect_validation_results("CASE001", GOOD_ENTITIES)
        score, label, hits = evaluate_rules("CASE001", validations, GOOD_ENTITIES)
        assert score == 0
        assert label == RiskLabel.GREEN
        assert len(hits) == 0

    def test_red_on_mismatch_case(self):
        validations = collect_validation_results("CASE002", MISMATCH_ENTITIES)
        score, label, hits = evaluate_rules("CASE002", validations, MISMATCH_ENTITIES)
        assert score > 30
        # With owner name + property_id + area mismatches → should be Red
        assert label in (RiskLabel.YELLOW, RiskLabel.RED)

    def test_mandatory_review_triggered(self):
        validations = collect_validation_results("CASE003", MISSING_ENTITIES)
        score, label, hits = evaluate_rules("CASE003", validations, MISSING_ENTITIES)
        mandatory = any(h.mandatory_review for h in hits)
        assert mandatory is True

    def test_low_confidence_rule(self):
        entities_low_conf = {
            "mother_deed": {
                "buyer_name": "Test User",
                "seller_name": "Seller",
                "property_id": "SY1",
                "area": "500 sqft",
                "extraction_confidence": {"owner_name": 0.50},   # below threshold
            },
            "khata": {
                "owner_name": "Test User",
                "property_id": "SY1",
                "area": "500 sqft",
            },
        }
        validations = collect_validation_results("CASE004", entities_low_conf)
        score, label, hits = evaluate_rules("CASE004", validations, entities_low_conf)
        hit_ids = [h.rule_id for h in hits]
        assert "LOW_CONFIDENCE_CRITICAL_FIELD" in hit_ids

    def test_score_capped_at_100(self):
        validations = collect_validation_results("CASE005", MISSING_ENTITIES)
        score, _, _ = evaluate_rules("CASE005", validations, MISSING_ENTITIES)
        assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# run_rules_pipeline (integration)
# ---------------------------------------------------------------------------

class TestRunRulesPipeline:
    def test_full_pipeline_good(self, tmp_path):
        output = run_rules_pipeline(
            case_id="INT001",
            case_entities=GOOD_ENTITIES,
            validations=collect_validation_results("INT001", GOOD_ENTITIES),
            persist=True,
            output_dir=tmp_path,
        )
        assert output.risk_label == RiskLabel.GREEN
        assert output.risk_score == 0
        assert (tmp_path / "INT001_risk.json").exists()

    def test_full_pipeline_mismatch(self, tmp_path):
        validations = collect_validation_results("INT002", MISMATCH_ENTITIES)
        output = run_rules_pipeline(
            case_id="INT002",
            case_entities=MISMATCH_ENTITIES,
            validations=validations,
            persist=True,
            output_dir=tmp_path,
        )
        assert output.risk_score > 30
        assert output.summary is not None

    def test_persist_creates_json(self, tmp_path):
        validations = collect_validation_results("INT003", GOOD_ENTITIES)
        run_rules_pipeline(
            case_id="INT003",
            case_entities=GOOD_ENTITIES,
            validations=validations,
            persist=True,
            output_dir=tmp_path,
        )
        import json
        out = json.loads((tmp_path / "INT003_risk.json").read_text())
        assert out["case_id"] == "INT003"
        assert "risk_label" in out
        assert "rule_hits" in out