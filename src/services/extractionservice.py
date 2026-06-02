"""
extractionservice.py
--------------------
Business-logic layer for entity extraction and normalisation.

Functions
---------
extract_entities(doc_type, ocr_blocks, page)  -> List[ExtractedEntity]
normalize_entity(entity)                      -> ExtractedEntity
persist_entities(case_id, document_id, source_doc, entities, db)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone

import spacy
import yaml
from dateutil import parser as dateutil_parser

from src.core.db_models import ExtractedEntityORM
from src.core.models import ExtractedEntity, OCRBlock

logger = logging.getLogger(__name__)


# ─── Load config ──────────────────────────────────────────────────────────────

def _load_config(path: str = "config/extraction_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

_CFG: dict = _load_config()


# ─── Lazy spaCy singleton ─────────────────────────────────────────────────────

_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        model = _CFG.get("spacy_model", "en_core_web_sm")
        try:
            _NLP = spacy.load(model)
        except OSError:
            logger.warning("spaCy model '%s' not found; falling back to blank 'en'.", model)
            _NLP = spacy.blank("en")
    return _NLP


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _avg_block_confidence(blocks: list[OCRBlock]) -> float:
    if not blocks:
        return 1.0
    return sum(b.confidence for b in blocks) / len(blocks)


def _confidence_with_ocr_penalty(base: float, ocr_blocks: list[OCRBlock]) -> float:
    avg_ocr   = _avg_block_confidence(ocr_blocks)
    threshold = _CFG["ocr_confidence_penalty_threshold"]
    penalty   = _CFG["ocr_confidence_penalty"]
    if avg_ocr < threshold:
        base = max(0.0, base - penalty)
    return round(min(base, 1.0), 4)


def _full_text(blocks: list[OCRBlock]) -> str:
    return "\n".join(b.text for b in blocks)


def _first_bbox(blocks: list[OCRBlock]) -> list[float]:
    return blocks[0].bbox if blocks else []


def _make_entity(
    field_name: str,
    value: str,
    confidence: float,
    doc_type: str,
    blocks: list[OCRBlock],
    page: int,
    method: str = "regex",
) -> ExtractedEntity:
    return ExtractedEntity(
        entity_id=str(uuid.uuid4()),
        case_id="",
        document_id="",
        doc_type=doc_type,
        field_name=field_name,
        value=value,
        normalized_value=value,        # overwritten by normalize_entity()
        confidence=_confidence_with_ocr_penalty(confidence, blocks),
        page=page,
        bbox=_first_bbox(blocks),
        source_doc="",
        extraction_method=method,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _regex_first(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def _extract_person_names(text: str, nlp) -> list[str]:
    doc = nlp(text[:5000])
    return list({ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"})


# ─── Mother Deed extractor ────────────────────────────────────────────────────

_SELLER_KW = re.compile(
    r"(?:vendor|seller|executant|transferor|mortgagor)[\:\s]+([A-Z][A-Za-z\s.]+?)(?:,|son|d/o|w/o|\n|aged)",
    re.IGNORECASE,
)
_BUYER_KW = re.compile(
    r"(?:vendee|buyer|purchaser|transferee|mortgagee)[\:\s]+([A-Z][A-Za-z\s.]+?)(?:,|son|d/o|w/o|\n|aged)",
    re.IGNORECASE,
)


def _extract_motherdeed(blocks: list[OCRBlock], page: int) -> list[ExtractedEntity]:
    cfg = _CFG["patterns"]
    nlp = _get_nlp()
    entities: list[ExtractedEntity] = []
    full = _full_text(blocks)

    # seller
    m = _SELLER_KW.search(full)
    if m:
        entities.append(_make_entity("seller_name", m.group(1).strip(), 0.80, "motherdeed", blocks, page))
    else:
        names = _extract_person_names(full, nlp)
        if names:
            entities.append(_make_entity("seller_name", names[0], 0.55, "motherdeed", blocks, page, "spacy"))

    # buyer
    m = _BUYER_KW.search(full)
    if m:
        entities.append(_make_entity("buyer_name", m.group(1).strip(), 0.80, "motherdeed", blocks, page))
    else:
        names = _extract_person_names(full, nlp)
        if len(names) >= 2:
            entities.append(_make_entity("buyer_name", names[1], 0.50, "motherdeed", blocks, page, "spacy"))

    # dates
    date_pats = cfg["date"]["written"] + "|" + cfg["date"]["dmy"] + "|" + cfg["date"]["iso"]
    for m in re.finditer(date_pats, full, re.IGNORECASE):
        raw     = m.group(0).strip()
        context = full[max(0, m.start()-40):m.start()].lower()
        if "execution" in context:
            entities.append(_make_entity("execution_date", raw, 0.82, "motherdeed", blocks, page))
        elif "registr" in context:
            entities.append(_make_entity("registration_date", raw, 0.82, "motherdeed", blocks, page))

    # survey / site / flat numbers
    for field, key in [
        ("survey_number", "survey_number"),
        ("site_number",   "site_number"),
        ("flat_number",   "flat_number"),
    ]:
        hit = _regex_first(cfg[key], full)
        if hit:
            entities.append(_make_entity(field, hit, 0.88, "motherdeed", blocks, page))

    # document number
    hit = _regex_first(cfg["document_number"], full)
    if hit:
        entities.append(_make_entity("document_number", hit, 0.90, "motherdeed", blocks, page))

    # registration office
    hit = _regex_first(cfg["registration_office"], full)
    if hit:
        entities.append(_make_entity("registration_office", hit, 0.78, "motherdeed", blocks, page))

    # property description (between Schedule-of-Property and Boundaries/Measurements)
    sched = re.search(
        r"Schedule(?:\s+of\s+Property)?[\:\s\-]+(.{50,400}?)(?=\n\n|\Z|Boundaries|Measurements)",
        full, re.IGNORECASE | re.DOTALL,
    )
    if sched:
        entities.append(_make_entity("property_description", sched.group(1).strip(), 0.70, "motherdeed", blocks, page))

    return entities


# ─── Khata extractor ──────────────────────────────────────────────────────────

_OWNER_KW = re.compile(
    r"(?:owner['\u2019\u2018`]?s?\s+name|name\s+of\s+(?:the\s+)?owner|owner)\s*[:\-]\s*([A-Za-z][A-Za-z\s.]{3,60}?)(?=\s*(?:,|\r?\n|son\b|d/o|w/o|khata|ward|zone|$))",
    re.IGNORECASE,
)


def _extract_khata(blocks: list[OCRBlock], page: int) -> list[ExtractedEntity]:
    cfg = _CFG["patterns"]
    nlp = _get_nlp()
    entities: list[ExtractedEntity] = []
    full = _full_text(blocks)

    # owner — regex first
    m = _OWNER_KW.search(full)
    if m:
        entities.append(_make_entity("owner_name", m.group(1).strip(), 0.82, "khata", blocks, page))
    else:
        nlp_doc = nlp(full[:5000])
        # filter out known org tokens and short strings
        org_tokens = {ent.text.lower() for ent in nlp_doc.ents if ent.label_ == "ORG"}
        names = [
            ent.text.strip()
            for ent in nlp_doc.ents
            if ent.label_ == "PERSON"
            and ent.text.strip().lower() not in org_tokens
            and len(ent.text.strip().split()) >= 2   # at least first + last name
        ]
        if names:
            entities.append(_make_entity("owner_name", names[0], 0.55, "khata", blocks, page, "spacy"))

    # khata number
    hit = _regex_first(cfg["khata_number"], full)
    if hit:
        entities.append(_make_entity("khata_number", hit, 0.90, "khata", blocks, page))

    # property id
    m = re.search(r"(?:Property\s+ID|PID)[\:\s#]*(\d[\d/\-]+)", full, re.IGNORECASE)
    if m:
        entities.append(_make_entity("property_id", m.group(1).strip(), 0.88, "khata", blocks, page))

    # ward / zone / assessment / usage
    for field, key in [("ward","ward"), ("zone","zone"), ("assessment","assessment"), ("usage","usage")]:
        hit = _regex_first(cfg[key], full)
        if hit:
            entities.append(_make_entity(field, hit, 0.85, "khata", blocks, page))

    # address
    addr = re.search(
        r"(?:Property\s+)?Address[\:\s]+(.{20,200}?)(?=\n\n|\Z|Ward|Zone|Khata)",
        full, re.IGNORECASE | re.DOTALL,
    )
    if addr:
        entities.append(_make_entity("address", addr.group(1).strip(), 0.72, "khata", blocks, page))

    return entities


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_entities(
    doc_type: str,
    ocr_blocks: list[OCRBlock],
    page: int = 0,
) -> list[ExtractedEntity]:
    """
    Extract + normalise entities from one page of OCR blocks.

    Parameters
    ----------
    doc_type   : "motherdeed" or "khata"
    ocr_blocks : list of OCRBlock (from OCR pipeline output)
    page       : 0-based page index

    Returns list of normalised ExtractedEntity objects.
    """
    if not ocr_blocks:
        return []
    if doc_type == "motherdeed":
        raw = _extract_motherdeed(ocr_blocks, page)
    elif doc_type == "khata":
        raw = _extract_khata(ocr_blocks, page)
    else:
        logger.warning("Unknown doc_type '%s'; returning empty list.", doc_type)
        return []
    return [normalize_entity(e) for e in raw]


def normalize_entity(entity: ExtractedEntity) -> ExtractedEntity:
    """
    Apply normalisation and set entity.normalized_value:
      Names  → title-case + collapsed whitespace
      Dates  → ISO 8601 (YYYY-MM-DD) via dateutil
      IDs    → trimmed, uppercase, spaces stripped
    Returns the mutated entity.
    """
    v = entity.value.strip()

    if entity.field_name in {"seller_name","buyer_name","owner_name","registration_office"}:
        entity.normalized_value = " ".join(v.title().split())

    elif entity.field_name in {"execution_date","registration_date"}:
        try:
            entity.normalized_value = dateutil_parser.parse(v, dayfirst=True).strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            entity.normalized_value = v

    elif entity.field_name in {"survey_number","site_number","flat_number",
                                "document_number","khata_number","property_id"}:
        entity.normalized_value = re.sub(r"\s+", "", v).upper()

    elif entity.field_name == "usage":
        entity.normalized_value = v.strip().capitalize()

    elif entity.field_name in {"ward","zone","assessment"}:
        entity.normalized_value = " ".join(v.split())

    else:
        entity.normalized_value = v

    return entity


def persist_entities(
    case_id: str,
    document_id: str,
    source_doc: str,
    entities: list[ExtractedEntity],
    db,                                  # SQLAlchemy Session
) -> None:
    """Upsert extracted entities into extracted_entities table."""
    for e in entities:
        e.case_id     = case_id
        e.document_id = document_id
        e.source_doc  = source_doc

        orm = ExtractedEntityORM(
            id                = e.entity_id,
            case_id           = e.case_id,
            document_id       = e.document_id,
            doc_type          = e.doc_type,
            field_name        = e.field_name,
            value             = e.value,
            normalized_value  = e.normalized_value,
            confidence        = str(round(e.confidence, 4)),
            page              = e.page,
            bbox              = json.dumps(e.bbox),
            source_doc        = e.source_doc,
            extraction_method = e.extraction_method,
            created_at        = datetime.now(timezone.utc),
        )
        db.merge(orm)

    db.commit()
    logger.info("Persisted %d entities for case=%s doc=%s", len(entities), case_id, document_id)