"""
tests/test_phase6_agents.py
Phase 6 smoke tests – Agentic AI Layer
"""

from __future__ import annotations

from src.core.validation_models import RiskLabel
from src.services import agent_service
from src.services.agent_service import (
    AgentOutputs,
    run_agents,
    run_compliance_agent,
    run_extraction_validator,
    run_khata_analysis_agent,
    run_risk_synthesis_agent,
    run_title_chain_agent,
)


def _entity(doc_type: str, field_name: str, value: str, confidence: float = 0.95) -> dict:
    return {
        "entity_id": f"ent_{doc_type}_{field_name}",
        "case_id": "case_test_001",
        "document_id": f"doc_{doc_type}",
        "doc_type": doc_type,
        "field_name": field_name,
        "value": value,
        "normalized_value": value.lower().strip(),
        "confidence": confidence,
        "page": 0,
    }


GOOD_ENTITIES = [
    _entity("motherdeed", "seller_name", "Ramesh Kumar"),
    _entity("motherdeed", "buyer_name", "Suresh Naik"),
    _entity("motherdeed", "execution_date", "2022-03-15"),
    _entity("motherdeed", "registration_date", "2022-03-20"),
    _entity("motherdeed", "survey_number", "123/4A"),
    _entity("motherdeed", "document_number", "BLR-2022-78901"),
    _entity("khata", "owner_name", "Suresh Naik"),
    _entity("khata", "khata_number", "KHT-5678"),
    _entity("khata", "property_id", "BBMP-001-2022"),
    _entity("khata", "ward", "Ward 42 Shivajinagar"),
    _entity("khata", "zone", "East Zone"),
    _entity("khata", "usage", "Residential"),
    _entity("khata", "assessment", "12000"),
]

MISMATCH_ENTITIES = [
    _entity("motherdeed", "seller_name", "Ramesh Kumar"),
    _entity("motherdeed", "buyer_name", "Arvind Sharma"),
    _entity("motherdeed", "execution_date", "2022-03-15"),
    _entity("motherdeed", "registration_date", "2022-03-20"),
    _entity("motherdeed", "survey_number", "123/4A"),
    _entity("motherdeed", "document_number", "BLR-2022-78901"),
    _entity("khata", "owner_name", "Suresh Naik"),
    _entity("khata", "khata_number", "KHT-5678"),
    _entity("khata", "property_id", "BBMP-001-2022"),
    _entity("khata", "ward", "Ward 42"),
    _entity("khata", "zone", "East Zone"),
]

SPARSE_ENTITIES = [
    _entity("motherdeed", "seller_name", "Ramesh Kumar", confidence=0.4),
    _entity("khata", "owner_name", "Unknown"),
]


class TestExtractionValidator:
    def test_complete_entities_high_score(self):
        out = run_extraction_validator(GOOD_ENTITIES, ocr_confidence=0.95)
        assert out.completeness_score >= 0.8
        assert len(out.missing_fields) == 0

    def test_sparse_entities_low_score(self):
        out = run_extraction_validator(SPARSE_ENTITIES, ocr_confidence=0.5)
        assert out.completeness_score < 0.5
        assert len(out.missing_fields) > 0
        assert len(out.recommendations) > 0

    def test_low_confidence_flag(self):
        out = run_extraction_validator(SPARSE_ENTITIES, ocr_confidence=0.95)
        assert "motherdeed.seller_name" in out.low_confidence_fields


class TestTitleChainAgent:
    def test_matching_names_ok(self):
        out = run_title_chain_agent(GOOD_ENTITIES)
        assert out.continuity_ok is True
        assert out.flag == "CHAIN_OK"

    def test_mismatched_names_broken(self):
        out = run_title_chain_agent(MISMATCH_ENTITIES)
        assert out.continuity_ok is False
        assert out.flag == "CHAIN_BROKEN"

    def test_missing_buyer_flag(self):
        entities_no_buyer = [e for e in GOOD_ENTITIES if e["field_name"] != "buyer_name"]
        out = run_title_chain_agent(entities_no_buyer)
        assert out.continuity_ok is False
        assert out.flag == "MISSING_BUYER"


class TestKhataAnalysisAgent:
    def test_residential_usage_detected(self):
        out = run_khata_analysis_agent(GOOD_ENTITIES)
        assert out.usage_classification == "residential"

    def test_ward_zone_extracted(self):
        out = run_khata_analysis_agent(GOOD_ENTITIES)
        assert out.ward is not None
        assert out.zone is not None

    def test_unknown_khata_type_noted(self):
        out = run_khata_analysis_agent(GOOD_ENTITIES)
        assert out.khata_type == "unknown"
        assert any("Khata type uncertain" in note for note in out.notes)


class TestComplianceAgent:
    def test_all_checks_pass_on_good_entities(self):
        tc = run_title_chain_agent(GOOD_ENTITIES)
        out = run_compliance_agent(GOOD_ENTITIES, title_chain_output=tc)
        assert out.all_passed is True
        assert out.failed_count == 0

    def test_checks_fail_on_sparse_entities(self):
        tc = run_title_chain_agent(SPARSE_ENTITIES)
        out = run_compliance_agent(SPARSE_ENTITIES, title_chain_output=tc)
        assert out.failed_count > 0

    def test_owner_name_inconsistency_fails(self):
        tc = run_title_chain_agent(MISMATCH_ENTITIES)
        out = run_compliance_agent(MISMATCH_ENTITIES, title_chain_output=tc)
        owner_check = next((c for c in out.checklist if c.check_id == "CHK_OWNER_NAME_CONSISTENT"), None)
        assert owner_check is not None
        assert owner_check.passed is False


class TestRiskSynthesisAgent:
    def test_low_risk_on_good_entities(self):
        ev = run_extraction_validator(GOOD_ENTITIES)
        tc = run_title_chain_agent(GOOD_ENTITIES)
        comp = run_compliance_agent(GOOD_ENTITIES, title_chain_output=tc)
        out = run_risk_synthesis_agent(
            extraction_validator=ev,
            title_chain=tc,
            compliance=comp,
            case_id="case_test_good",
        )
        assert out.risk_label == RiskLabel.GREEN
        assert out.risk_score < 30

    def test_high_risk_on_mismatch(self):
        ev = run_extraction_validator(SPARSE_ENTITIES)
        tc = run_title_chain_agent(MISMATCH_ENTITIES)
        comp = run_compliance_agent(MISMATCH_ENTITIES, title_chain_output=tc)
        out = run_risk_synthesis_agent(
            extraction_validator=ev,
            title_chain=tc,
            compliance=comp,
            case_id="case_test_bad",
        )
        assert out.risk_score > 30
        assert out.final_summary != ""


class TestRunAgentsOrchestrator:
    def test_returns_agent_outputs_type(self, monkeypatch):
        monkeypatch.setattr(
            agent_service,
            "generate_explanations_llm",
            lambda summary: "Mock explanation for testing.",
        )
        result = run_agents(case_id="case_orch_001", entities=GOOD_ENTITIES)
        assert isinstance(result, AgentOutputs)

    def test_all_agents_populated(self, monkeypatch):
        monkeypatch.setattr(
            agent_service,
            "generate_explanations_llm",
            lambda summary: "Mock explanation for testing.",
        )
        result = run_agents(case_id="case_orch_001", entities=GOOD_ENTITIES)
        assert result.extraction_validator is not None
        assert result.title_chain is not None
        assert result.khata_analysis is not None
        assert result.compliance is not None
        assert result.risk_synthesis is not None

    def test_no_errors_on_good_entities(self, monkeypatch):
        monkeypatch.setattr(
            agent_service,
            "generate_explanations_llm",
            lambda summary: "Mock explanation for testing.",
        )
        result = run_agents(case_id="case_orch_002", entities=GOOD_ENTITIES)
        assert len(result.errors) == 0

    def test_llm_explanations_generated(self, monkeypatch):
        monkeypatch.setattr(
            agent_service,
            "generate_explanations_llm",
            lambda summary: "Mock explanation for testing.",
        )
        result = run_agents(case_id="case_orch_003", entities=GOOD_ENTITIES)
        assert isinstance(result.llm_explanations, str)
        assert len(result.llm_explanations) > 0

    def test_empty_entities_no_crash(self, monkeypatch):
        monkeypatch.setattr(
            agent_service,
            "generate_explanations_llm",
            lambda summary: "Mock explanation for testing.",
        )
        result = run_agents(case_id="case_orch_empty", entities=[])
        assert result.extraction_validator is not None
        assert result.extraction_validator.completeness_score == 0.0