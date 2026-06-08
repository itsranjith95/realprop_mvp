"""
Unit Tests — Evidence Service
Tests for storing, fetching, and formatting evidence records.
"""
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file for each test."""
    db_file = tmp_path / "test_realprop.db"
    import src.services.evidence_service as ev_svc
    monkeypatch.setattr(ev_svc, "DB_PATH", db_file)
    return db_file, ev_svc


class TestEvidenceServiceStore:

    def test_store_evidence_returns_id(self, tmp_db):
        _, ev_svc = tmp_db
        eid = ev_svc.store_evidence(
            case_id="CASE_001",
            source_document_id="DOC_MD_001",
            rule_id="R010",
            rule_version="1.1.0",
            page=2,
            bbox=[10.0, 20.0, 100.0, 50.0],
            field_name="buyer_name",
            extracted_value="Ranjith Kumar",
            doc_type="MOTHER_DEED",
            rule_name="OWNER_NAME_MISMATCH",
            severity="high",
            flag_description="Buyer name mismatch",
        )
        assert isinstance(eid, int) and eid > 0

    def test_fetch_evidence_for_case(self, tmp_db):
        _, ev_svc = tmp_db
        ev_svc.store_evidence(
            case_id="CASE_002", source_document_id="DOC_KH_002",
            rule_id="R001", rule_version="1.0.0", severity="high",
            flag_description="Missing owner name",
        )
        records = ev_svc.fetch_evidence_for_case("CASE_002")
        assert len(records) == 1
        assert records[0]["rule_id"] == "R001"
        assert records[0]["case_id"] == "CASE_002"

    def test_fetch_evidence_returns_empty_for_unknown_case(self, tmp_db):
        _, ev_svc = tmp_db
        records = ev_svc.fetch_evidence_for_case("CASE_NONEXISTENT")
        assert records == []

    def test_fetch_evidence_for_rule(self, tmp_db):
        _, ev_svc = tmp_db
        for i in range(3):
            ev_svc.store_evidence(
                case_id="CASE_003", source_document_id=f"DOC_{i}",
                rule_id="R010", rule_version="1.1.0",
                page=i, severity="high",
            )
        ev_svc.store_evidence(
            case_id="CASE_003", source_document_id="DOC_X",
            rule_id="R011", rule_version="1.1.0", severity="high",
        )
        r010_records = ev_svc.fetch_evidence_for_rule("CASE_003", "R010")
        assert len(r010_records) == 3
        r011_records = ev_svc.fetch_evidence_for_rule("CASE_003", "R011")
        assert len(r011_records) == 1

    def test_store_evidence_batch(self, tmp_db):
        _, ev_svc = tmp_db
        records = [
            {"case_id": "CASE_BATCH", "source_document_id": "DOC_A",
             "rule_id": "R001", "rule_version": "1.0.0", "severity": "high"},
            {"case_id": "CASE_BATCH", "source_document_id": "DOC_B",
             "rule_id": "R002", "rule_version": "1.0.0", "severity": "high"},
        ]
        ids = ev_svc.store_evidence_batch(records)
        assert len(ids) == 2
        all_ev = ev_svc.fetch_evidence_for_case("CASE_BATCH")
        assert len(all_ev) == 2


class TestEvidenceServiceFormat:

    def test_format_evidence_for_ui_structure(self, tmp_db):
        _, ev_svc = tmp_db
        ev_svc.store_evidence(
            case_id="CASE_FMT", source_document_id="DOC_MD",
            rule_id="R010", rule_version="1.1.0",
            field_name="buyer_name", extracted_value="Ranjith Kumar",
            doc_type="MOTHER_DEED", rule_name="OWNER_NAME_MISMATCH",
            severity="high", page=1, bbox=[0, 0, 100, 50],
        )
        records = ev_svc.fetch_evidence_for_case("CASE_FMT")
        formatted = ev_svc.format_evidence_for_ui(records)
        assert len(formatted) == 1
        row = formatted[0]
        assert "Rule" in row
        assert "Severity" in row
        assert "Document" in row
        assert "HIGH" in row["Severity"]