"""
Unit tests – Phase 4 Entity Extraction & Normalisation
Run: pytest tests/unit/test_extraction_pipeline.py -v
"""

import pytest
from src.core.models import OCRBlock, ExtractedEntity
from src.services.extractionservice import extract_entities, normalize_entity


def make_block(text: str, conf: float = 0.95) -> OCRBlock:
    return OCRBlock(text=text, confidence=conf, bbox=[0, 0, 100, 20])


MOTHER_DEED_TEXT = "\n".join([
    "This Sale Deed is executed on 15th March 2022 between",
    "Vendor: Ramesh Kumar Sharma, Son of Late Mohan Lal Sharma,",
    "and Vendee: Surekha Vijay Nair, Wife of Vijay Nair.",
    "",
    "Schedule of Property:",
    "Sy. No. 45/3, measuring 1200 sq.ft, situated at Koramangala,",
    "Site No. 12, Bengaluru.",
    "",
    "Flat No. B-204, Registration on 16-03-2022",
    "Doc. No. 1234/2022",
    "Sub-Registrar Office, Koramangala, Bengaluru.",
])

KHATA_TEXT = "\n".join([
    "Khata No. 456/23",
    "Owner's Name: Surekha Vijay Nair",   # plain ASCII apostrophe
    "Property ID: 789012",
    "Ward No. 68 - Koramangala",
    "Zone: South Zone",
    "Usage: Residential",
    "Address: No. 12, 5th Cross, Koramangala, Bengaluru - 560034",
    "Annual Value Rs. 12,000",
    "BRUHAT BENGALURU MAHANAGARA PALIKE",
])


def _blocks(text: str) -> list[OCRBlock]:
    return [make_block(line.strip()) for line in text.splitlines() if line.strip()]

# Mother Deed tests
def test_motherdeed_extracts_entities():
    assert len(extract_entities("motherdeed", _blocks(MOTHER_DEED_TEXT))) > 0

def test_motherdeed_seller_name():
    sellers = [e for e in extract_entities("motherdeed", _blocks(MOTHER_DEED_TEXT)) if e.field_name == "seller_name"]
    assert len(sellers) >= 1
    assert "ramesh" in sellers[0].normalized_value.lower()

def test_motherdeed_date_normalised_to_iso():
    dates = [e for e in extract_entities("motherdeed", _blocks(MOTHER_DEED_TEXT)) if "date" in e.field_name]
    assert any(len(d.normalized_value) == 10 and d.normalized_value[4] == "-" for d in dates)

def test_motherdeed_survey_number():
    surveys = [e for e in extract_entities("motherdeed", _blocks(MOTHER_DEED_TEXT)) if e.field_name == "survey_number"]
    assert "45" in surveys[0].normalized_value

def test_motherdeed_confidence_range():
    for e in extract_entities("motherdeed", _blocks(MOTHER_DEED_TEXT)):
        assert 0.0 <= e.confidence <= 1.0

# Khata tests
def test_khata_owner_name():
    owners = [e for e in extract_entities("khata", _blocks(KHATA_TEXT)) if e.field_name == "owner_name"]
    assert "surekha" in owners[0].normalized_value.lower()

def test_khata_number_no_spaces():
    khata = [e for e in extract_entities("khata", _blocks(KHATA_TEXT)) if e.field_name == "khata_number"]
    assert " " not in khata[0].normalized_value

def test_khata_usage():
    usage = [e for e in extract_entities("khata", _blocks(KHATA_TEXT)) if e.field_name == "usage"]
    assert usage[0].normalized_value.lower() in ("residential","commercial","mixed","industrial")

# Normalisation unit tests
def test_normalize_date_iso():
    e = ExtractedEntity(entity_id="x",case_id="c",document_id="d",doc_type="motherdeed",
                        field_name="execution_date",value="15/03/2022",normalized_value="",confidence=0.9,page=0)
    assert normalize_entity(e).normalized_value == "2022-03-15"

def test_normalize_name_title_case():
    e = ExtractedEntity(entity_id="x",case_id="c",document_id="d",doc_type="motherdeed",
                        field_name="seller_name",value="RAMESH KUMAR SHARMA",normalized_value="",confidence=0.9,page=0)
    assert normalize_entity(e).normalized_value == "Ramesh Kumar Sharma"

def test_unknown_doc_type_returns_empty():
    assert extract_entities("unknown_type", _blocks("some text")) == []

def test_empty_blocks_returns_empty():
    assert extract_entities("motherdeed", []) == []