import json
from pathlib import Path

import pytest

from src.core.validation_models import ValidationResult
from src.pipelines.rules_pipeline import run_rules_pipeline


@pytest.fixture
def sample_validations():
    return [
        ValidationResult.model_validate({
            "rule_id": "R010",
            "type": "OWNER_NAME",
            "severity": "high",
            "passed": False,
            "description": "Buyer name in Mother Deed does not match owner name in Khata",
            "message": "Buyer name in Mother Deed does not match owner name in Khata",
            "details": {},
            "evidence": {
                "source_document_id": "DOC_MD_001",
                "page": 2,
                "bbox": [10, 20, 100, 40],
                "field_name": "buyer_name",
                "value": "Ranjith Kumar",
                "doc_type": "MOTHER_DEED",
                "rule_version": "1.1.0",
            },
        })
    ]


@pytest.fixture
def sample_entities():
    return {
        "mother_deed": {
            "buyer_name": {"value": "Ranjith Kumar", "confidence": 0.95},
            "seller_name": {"value": "Shyam Rao", "confidence": 0.95},
            "property_id": {"value": "SY-123/4A", "confidence": 0.95},
            "registration_date": {"value": "2023-08-15", "confidence": 0.95},
        },
        "khata": {
            "owner_name": {"value": "Rajesh Sharma", "confidence": 0.95},
            "khata_number": {"value": "KH-789", "confidence": 0.95},
            "property_id": {"value": "SY-123/4A", "confidence": 0.95},
        },
    }


def test_run_rules_pipeline_persists_output_and_evidence(
    tmp_path,
    monkeypatch,
    sample_validations,
    sample_entities,
):
    import src.services.evidence_service as ev_svc

    test_db = tmp_path / "test_realprop.db"
    monkeypatch.setattr(ev_svc, "DB_PATH", test_db)

    output_dir = tmp_path / "risk_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_output = run_rules_pipeline(
        case_id="CASE_INT_001",
        case_entities=sample_entities,
        validations=sample_validations,
        persist=True,
        output_dir=output_dir,
        attach_explanations=False,
    )

    assert risk_output.case_id == "CASE_INT_001"
    assert risk_output.risk_score >= 0
    assert len(risk_output.rule_hits) >= 1

    risk_file = output_dir / "CASE_INT_001_risk.json"
    assert risk_file.exists()

    payload = json.loads(risk_file.read_text(encoding="utf-8"))
    assert payload["case_id"] == "CASE_INT_001"
    assert "risk_score" in payload
    assert "rule_hits" in payload

    evidence_rows = ev_svc.fetch_evidence_for_case("CASE_INT_001")
    assert len(evidence_rows) >= 1
    assert evidence_rows[0]["case_id"] == "CASE_INT_001"
    assert evidence_rows[0]["rule_id"] == "R010"