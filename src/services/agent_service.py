"""
src/services/agent_service.py
Phase 6 – Agentic AI Layer

Five specialised agents run sequentially to produce AgentOutputs for a case.
An optional LLM helper generates natural-language explanations and a case-summary paragraph.

Agents are plain Python functions with typed Pydantic input/output –
no external orchestration framework is required for the MVP.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.core.validation_models import RiskLabel, RiskOutput, ValidationResult

logger = logging.getLogger(__name__)

# Load root .env explicitly
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class FieldStatus(BaseModel):
    field_name: str
    present: bool
    confidence: float = 0.0
    note: str = ""


class ExtractionValidatorOutput(BaseModel):
    """Output of the ExtractionValidator Agent."""
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of expected fields present",
    )
    missing_fields: list[str] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    field_statuses: list[FieldStatus] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class TitleChainOutput(BaseModel):
    """Output of the simplified Title Chain Agent (Mother Deed → Khata)."""
    continuity_ok: bool
    mother_deed_buyer: str | None = None
    khata_owner: str | None = None
    similarity_score: float = 0.0
    flag: str = ""
    notes: list[str] = Field(default_factory=list)


class KhataAnalysisOutput(BaseModel):
    """Output of the Khata Analysis Agent."""
    usage_classification: str = "unknown"
    ward: str | None = None
    zone: str | None = None
    khata_type: str = "unknown"
    notes: list[str] = Field(default_factory=list)


class ComplianceCheckItem(BaseModel):
    check_id: str
    description: str
    passed: bool
    severity: str = "medium"
    evidence: dict[str, Any] = Field(default_factory=dict)


class ComplianceAgentOutput(BaseModel):
    """Output of the Compliance Agent (Bengaluru/Karnataka basic checks)."""
    checklist: list[ComplianceCheckItem] = Field(default_factory=list)
    all_passed: bool = False
    failed_count: int = 0


class RiskSynthesisOutput(BaseModel):
    """Output of the Risk Synthesis Agent – shown to the lawyer."""
    risk_score: int = Field(ge=0, le=100)
    risk_label: RiskLabel
    mandatory_review: bool = False
    explanation_tree: list[dict[str, Any]] = Field(default_factory=list)
    final_summary: str = ""


class AgentOutputs(BaseModel):
    """Bundled result returned by run_agents()."""
    case_id: str
    extraction_validator: ExtractionValidatorOutput | None = None
    title_chain: TitleChainOutput | None = None
    khata_analysis: KhataAnalysisOutput | None = None
    compliance: ComplianceAgentOutput | None = None
    risk_synthesis: RiskSynthesisOutput | None = None
    llm_explanations: str = ""
    errors: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MOTHER_DEED_REQUIRED = [
    "seller_name",
    "buyer_name",
    "execution_date",
    "registration_date",
    "survey_number",
    "document_number",
]

KHATA_REQUIRED = [
    "owner_name",
    "khata_number",
    "property_id",
    "ward",
    "zone",
]

LOW_CONF_THRESHOLD = 0.60


def _normalise_name(name: str | None) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def _token_similarity(a: str, b: str) -> float:
    """Simple token-overlap ratio between two name strings."""
    if not a or not b:
        return 0.0
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / max(len(set_a), len(set_b))


def _entities_by_doc_type(
    entities: list[dict[str, Any]],
    doc_type: str,
) -> dict[str, dict[str, Any]]:
    """Return {field_name: entity_dict} for a given doc_type."""
    result: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if entity.get("doc_type", "") == doc_type:
            field_name = entity.get("field_name")
            if field_name:
                result[field_name] = entity
    return result


def _safe_message_content(data: dict[str, Any]) -> str:
    """
    Extract assistant message content safely from OpenAI-compatible chat response.
    Returns empty string if not found.
    """
    try:
        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
            return "\n".join(text_parts).strip()

        return ""
    except Exception:
        return ""


def _preferred_openrouter_model() -> str:
    """
    Allow override via .env, otherwise use a sensible default.
    """
    return os.getenv(
        "OPENROUTER_MODEL",
        "mistralai/mistral-small-3.2-24b-instruct:free",
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 – ExtractionValidator
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction_validator(
    entities: list[dict[str, Any]],
    ocr_confidence: float = 1.0,
) -> ExtractionValidatorOutput:
    """
    Checks completeness and confidence of extracted entities.

    Input:
        entities       – list of ExtractedEntity-like dicts
        ocr_confidence – overall page confidence from OCR (0–1)
    Output:
        ExtractionValidatorOutput
    """
    md_entities = _entities_by_doc_type(entities, "motherdeed")
    kh_entities = _entities_by_doc_type(entities, "khata")

    field_statuses: list[FieldStatus] = []
    missing: list[str] = []
    low_conf: list[str] = []

    all_required = [(field, "motherdeed") for field in MOTHER_DEED_REQUIRED] + [
        (field, "khata") for field in KHATA_REQUIRED
    ]

    for field, dtype in all_required:
        lookup = md_entities if dtype == "motherdeed" else kh_entities
        entity = lookup.get(field)

        if entity is None:
            field_statuses.append(
                FieldStatus(
                    field_name=f"{dtype}.{field}",
                    present=False,
                    confidence=0.0,
                    note="missing",
                )
            )
            missing.append(f"{dtype}.{field}")
        else:
            conf = float(entity.get("confidence", 1.0))
            note = "ok" if conf >= LOW_CONF_THRESHOLD else "low_confidence"
            field_statuses.append(
                FieldStatus(
                    field_name=f"{dtype}.{field}",
                    present=True,
                    confidence=conf,
                    note=note,
                )
            )
            if conf < LOW_CONF_THRESHOLD:
                low_conf.append(f"{dtype}.{field}")

    total = len(all_required)
    present_count = total - len(missing)
    completeness = round(present_count / total, 3) if total else 1.0

    recommendations: list[str] = []
    if missing:
        recommendations.append(f"Re-upload or manually provide: {', '.join(missing)}")
    if low_conf:
        recommendations.append(f"Consider re-OCR for low-confidence fields: {', '.join(low_conf)}")
    if ocr_confidence < 0.6:
        recommendations.append(
            "Overall OCR confidence is low — consider scanning the document at higher DPI."
        )

    return ExtractionValidatorOutput(
        completeness_score=completeness,
        missing_fields=missing,
        low_confidence_fields=low_conf,
        field_statuses=field_statuses,
        recommendations=recommendations,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 – Title Chain (simplified MVP)
# ─────────────────────────────────────────────────────────────────────────────

def run_title_chain_agent(
    entities: list[dict[str, Any]],
    similarity_threshold: float = 0.60,
) -> TitleChainOutput:
    """
    Checks that the Mother Deed buyer is the current Khata owner.
    Single-step title chain for MVP.
    """
    md = _entities_by_doc_type(entities, "motherdeed")
    kh = _entities_by_doc_type(entities, "khata")

    buyer_raw = md.get("buyer_name", {}).get("normalized_value") or md.get("buyer_name", {}).get("value")
    owner_raw = kh.get("owner_name", {}).get("normalized_value") or kh.get("owner_name", {}).get("value")

    buyer_norm = _normalise_name(buyer_raw)
    owner_norm = _normalise_name(owner_raw)

    notes: list[str] = []

    if not buyer_norm:
        notes.append("Mother Deed buyer_name not found — cannot verify title continuity.")
        return TitleChainOutput(
            continuity_ok=False,
            mother_deed_buyer=buyer_raw,
            khata_owner=owner_raw,
            similarity_score=0.0,
            flag="MISSING_BUYER",
            notes=notes,
        )

    if not owner_norm:
        notes.append("Khata owner_name not found — cannot verify title continuity.")
        return TitleChainOutput(
            continuity_ok=False,
            mother_deed_buyer=buyer_raw,
            khata_owner=owner_raw,
            similarity_score=0.0,
            flag="MISSING_OWNER",
            notes=notes,
        )

    sim = _token_similarity(buyer_norm, owner_norm)
    ok = sim >= similarity_threshold

    if ok:
        flag = "CHAIN_OK"
        notes.append(
            f"Title chain intact. Buyer '{buyer_raw}' matches Khata owner '{owner_raw}' "
            f"(similarity={sim:.2f})."
        )
    else:
        flag = "CHAIN_BROKEN"
        notes.append(
            f"Title continuity broken: Mother Deed buyer '{buyer_raw}' does not match "
            f"Khata owner '{owner_raw}' (similarity={sim:.2f}, threshold={similarity_threshold}). "
            "Manual review required."
        )

    return TitleChainOutput(
        continuity_ok=ok,
        mother_deed_buyer=buyer_raw,
        khata_owner=owner_raw,
        similarity_score=round(sim, 3),
        flag=flag,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 – Khata Analysis
# ─────────────────────────────────────────────────────────────────────────────

_USAGE_KEYWORDS: dict[str, list[str]] = {
    "residential": ["residential", "dwelling", "house", "flat", "apartment", "villa"],
    "commercial": ["commercial", "shop", "office", "showroom", "godown", "warehouse"],
    "mixed": ["mixed", "combined", "partly"],
    "vacant": ["vacant", "open", "site", "plot"],
}

_KHATA_TYPE_KEYWORDS: dict[str, list[str]] = {
    "A-Khata": ["a khata", "a-khata", "bbmp a", "regular khata"],
    "B-Khata": ["b khata", "b-khata", "bbmp b", "revenue site"],
}


def run_khata_analysis_agent(
    entities: list[dict[str, Any]],
) -> KhataAnalysisOutput:
    """
    Classifies Khata usage, extracts ward/zone, and attempts Khata-type detection.
    """
    kh = _entities_by_doc_type(entities, "khata")
    notes: list[str] = []

    usage_raw = (kh.get("usage", {}).get("value") or "").lower()
    usage_class = "unknown"
    for label, keywords in _USAGE_KEYWORDS.items():
        if any(keyword in usage_raw for keyword in keywords):
            usage_class = label
            break

    if usage_class == "unknown" and usage_raw:
        notes.append(f"Usage field present but classification unclear: '{usage_raw}'")

    ward = kh.get("ward", {}).get("normalized_value") or kh.get("ward", {}).get("value")
    zone = kh.get("zone", {}).get("normalized_value") or kh.get("zone", {}).get("value")

    all_values = " ".join(
        (value.get("value", "") + " " + value.get("normalized_value", ""))
        for value in kh.values()
    ).lower()

    khata_type = "unknown"
    for ktype, keywords in _KHATA_TYPE_KEYWORDS.items():
        if any(keyword in all_values for keyword in keywords):
            khata_type = ktype
            break

    if khata_type == "unknown":
        notes.append("Khata type uncertain — could not determine A-Khata or B-Khata from text.")
    if not ward:
        notes.append("Ward not found in Khata.")
    if not zone:
        notes.append("Zone not found in Khata.")

    return KhataAnalysisOutput(
        usage_classification=usage_class,
        ward=ward,
        zone=zone,
        khata_type=khata_type,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 – Compliance (Bengaluru-specific)
# ─────────────────────────────────────────────────────────────────────────────

_BENGALURU_RULES: list[dict[str, Any]] = [
    {
        "check_id": "CHK_KHATA_PRESENT",
        "description": "Khata document must be present for the property",
        "severity": "critical",
    },
    {
        "check_id": "CHK_MOTHERDEED_PRESENT",
        "description": "Mother Deed document must be present",
        "severity": "critical",
    },
    {
        "check_id": "CHK_OWNER_NAME_CONSISTENT",
        "description": "Owner name must be consistent across Mother Deed (buyer) and Khata",
        "severity": "high",
    },
    {
        "check_id": "CHK_SURVEY_NUMBER_PRESENT",
        "description": "Survey/Site number must be present in Mother Deed",
        "severity": "medium",
    },
    {
        "check_id": "CHK_REGISTRATION_DATE_PRESENT",
        "description": "Registration date must be present in Mother Deed",
        "severity": "medium",
    },
    {
        "check_id": "CHK_KHATA_NUMBER_PRESENT",
        "description": "Khata number must be present in Khata document",
        "severity": "high",
    },
    {
        "check_id": "CHK_WARD_ZONE_PRESENT",
        "description": "Ward and Zone must be present in Khata (BBMP jurisdiction verification)",
        "severity": "medium",
    },
]


def run_compliance_agent(
    entities: list[dict[str, Any]],
    title_chain_output: TitleChainOutput | None = None,
) -> ComplianceAgentOutput:
    """
    Runs Bengaluru/Karnataka-specific compliance checks.
    """
    md = _entities_by_doc_type(entities, "motherdeed")
    kh = _entities_by_doc_type(entities, "khata")

    checklist: list[ComplianceCheckItem] = []

    for rule in _BENGALURU_RULES:
        cid = rule["check_id"]
        passed = False
        evidence: dict[str, Any] = {}

        if cid == "CHK_KHATA_PRESENT":
            passed = bool(kh)
            evidence = {"khata_fields_found": list(kh.keys())}

        elif cid == "CHK_MOTHERDEED_PRESENT":
            passed = bool(md)
            evidence = {"motherdeed_fields_found": list(md.keys())}

        elif cid == "CHK_OWNER_NAME_CONSISTENT":
            if title_chain_output:
                passed = title_chain_output.continuity_ok
                evidence = {
                    "mother_deed_buyer": title_chain_output.mother_deed_buyer,
                    "khata_owner": title_chain_output.khata_owner,
                    "similarity_score": title_chain_output.similarity_score,
                }
            else:
                buyer = md.get("buyer_name", {}).get("normalized_value", "")
                owner = kh.get("owner_name", {}).get("normalized_value", "")
                sim = _token_similarity(_normalise_name(buyer), _normalise_name(owner))
                passed = sim >= 0.6
                evidence = {"buyer": buyer, "owner": owner, "similarity": sim}

        elif cid == "CHK_SURVEY_NUMBER_PRESENT":
            survey = md.get("survey_number") or md.get("site_number")
            passed = survey is not None and bool(survey.get("value"))
            evidence = {"value": survey.get("value") if survey else None}

        elif cid == "CHK_REGISTRATION_DATE_PRESENT":
            registration_date = md.get("registration_date")
            passed = registration_date is not None and bool(registration_date.get("value"))
            evidence = {"value": registration_date.get("value") if registration_date else None}

        elif cid == "CHK_KHATA_NUMBER_PRESENT":
            khata_number = kh.get("khata_number")
            passed = khata_number is not None and bool(khata_number.get("value"))
            evidence = {"value": khata_number.get("value") if khata_number else None}

        elif cid == "CHK_WARD_ZONE_PRESENT":
            ward = kh.get("ward")
            zone = kh.get("zone")
            passed = (
                ward is not None and bool(ward.get("value")) and
                zone is not None and bool(zone.get("value"))
            )
            evidence = {
                "ward": ward.get("value") if ward else None,
                "zone": zone.get("value") if zone else None,
            }

        checklist.append(
            ComplianceCheckItem(
                check_id=cid,
                description=rule["description"],
                passed=passed,
                severity=rule["severity"],
                evidence=evidence,
            )
        )

    failed = [item for item in checklist if not item.passed]
    return ComplianceAgentOutput(
        checklist=checklist,
        all_passed=len(failed) == 0,
        failed_count=len(failed),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 – Risk Synthesis
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_WEIGHT: dict[str, int] = {
    "critical": 30,
    "high": 15,
    "medium": 8,
    "low": 3,
    "info": 0,
}


def run_risk_synthesis_agent(
    rule_outputs: list[ValidationResult] | None = None,
    risk_output: RiskOutput | None = None,
    extraction_validator: ExtractionValidatorOutput | None = None,
    title_chain: TitleChainOutput | None = None,
    compliance: ComplianceAgentOutput | None = None,
    case_id: str = "unknown",
) -> RiskSynthesisOutput:
    """
    Synthesises all agent and rule findings into a final risk score + explanation tree.
    """
    mandatory_review = False
    explanation_tree: list[dict[str, Any]] = []

    base_score = risk_output.risk_score if risk_output else 0
    if risk_output and getattr(risk_output, "mandatory_review", False):
        mandatory_review = True

    agent_score_delta = 0

    if title_chain and not title_chain.continuity_ok:
        flag = title_chain.flag
        delta = 20 if flag == "CHAIN_BROKEN" else 10
        agent_score_delta += delta
        mandatory_review = True
        explanation_tree.append(
            {
                "source": "TitleChainAgent",
                "flag": flag,
                "delta": delta,
                "detail": title_chain.notes,
            }
        )

    if extraction_validator:
        missing_count = len(extraction_validator.missing_fields)
        delta = min(missing_count * 5, 20)
        agent_score_delta += delta
        if missing_count > 0:
            explanation_tree.append(
                {
                    "source": "ExtractionValidatorAgent",
                    "completeness_score": extraction_validator.completeness_score,
                    "missing_fields": extraction_validator.missing_fields,
                    "delta": delta,
                    "detail": extraction_validator.recommendations,
                }
            )

    if compliance:
        for item in compliance.checklist:
            if not item.passed:
                weight = _SEVERITY_WEIGHT.get(item.severity, 8)
                agent_score_delta += weight
                if item.severity == "critical":
                    mandatory_review = True
                explanation_tree.append(
                    {
                        "source": "ComplianceAgent",
                        "check_id": item.check_id,
                        "description": item.description,
                        "severity": item.severity,
                        "delta": weight,
                        "evidence": item.evidence,
                    }
                )

    score = min(base_score + agent_score_delta, 100)

    if score >= 60:
        label = RiskLabel.RED
        mandatory_review = True
    elif score >= 30:
        label = RiskLabel.YELLOW
    else:
        label = RiskLabel.GREEN

    parts: list[str] = []
    if label == RiskLabel.RED:
        parts.append(f"Case {case_id} is HIGH RISK (score={score}).")
    elif label == RiskLabel.YELLOW:
        parts.append(f"Case {case_id} has MODERATE RISK (score={score}).")
    else:
        parts.append(f"Case {case_id} appears LOW RISK (score={score}).")

    if mandatory_review:
        parts.append("Manual legal review is MANDATORY.")
    if title_chain and not title_chain.continuity_ok:
        parts.append(f"Title chain issue: {'; '.join(title_chain.notes)}")
    if extraction_validator and extraction_validator.missing_fields:
        parts.append(f"Missing document fields: {', '.join(extraction_validator.missing_fields)}.")
    if compliance and compliance.failed_count > 0:
        failed_ids = [item.check_id for item in compliance.checklist if not item.passed]
        parts.append(f"Failed compliance checks: {', '.join(failed_ids)}.")

    return RiskSynthesisOutput(
        risk_score=score,
        risk_label=label,
        mandatory_review=mandatory_review,
        explanation_tree=explanation_tree,
        final_summary=" ".join(parts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-Assisted Tasks
# ─────────────────────────────────────────────────────────────────────────────

def _build_structured_summary(s: dict[str, Any]) -> str:
    """
    Builds a structured, section-by-section lawyer-facing summary.
    Used both as the LLM prompt context and as the deterministic fallback output.
    """
    case_id = s.get("case_id", "unknown")
    risk_score = s.get("risk_score", 0)
    risk_label = s.get("risk_label", "Unknown")
    mandatory = s.get("mandatory_review", False)
    missing = s.get("missing_fields", [])
    flags = s.get("flags", [])
    title_ok = s.get("title_continuity_ok", None)
    title_notes = s.get("title_continuity_notes", [])
    compliance_failed = s.get("compliance_failed", [])
    compliance_passed = s.get("compliance_passed", [])

    lines: list[str] = []

    lines.append("EXECUTIVE SUMMARY")
    lines.append("─" * 50)
    if risk_label == "Green":
        lines.append(
            f"Case {case_id} has been assessed as LOW RISK (Score: {risk_score}/100). "
            "Documents appear complete and consistent. Proceed with standard verification."
        )
    elif risk_label == "Yellow":
        lines.append(
            f"Case {case_id} has been assessed as MODERATE RISK (Score: {risk_score}/100). "
            "Certain mandatory fields or compliance checks are incomplete. "
            "Legal counsel should review before proceeding."
        )
    else:
        lines.append(
            f"Case {case_id} has been assessed as HIGH RISK (Score: {risk_score}/100). "
            "Significant document deficiencies or title chain issues were identified. "
            "Do NOT proceed without full legal review."
        )
    lines.append("")

    critical_flags = [
        f for f in flags
        if "critical" in f.lower() or "broken" in f.lower() or "missing" in f.lower()
    ]
    lines.append("CRITICAL ISSUES")
    lines.append("─" * 50)
    if critical_flags:
        for flag in critical_flags:
            lines.append(f"  ✗  {flag}")
    elif risk_label == "Green":
        lines.append("  ✓  No critical issues identified.")
    else:
        lines.append("  ✗  One or more compliance checks failed (see Compliance Findings below).")
    lines.append("")

    lines.append("MISSING DOCUMENT FIELDS")
    lines.append("─" * 50)
    if missing:
        md_missing = [
            f.replace("motherdeed.", "Mother Deed → ")
            for f in missing
            if f.startswith("motherdeed")
        ]
        kh_missing = [
            f.replace("khata.", "Khata → ")
            for f in missing
            if f.startswith("khata")
        ]

        if md_missing:
            lines.append("  Mother Deed:")
            for field in md_missing:
                lines.append(f"    ✗  {field}")

        if kh_missing:
            lines.append("  Khata:")
            for field in kh_missing:
                lines.append(f"    ✗  {field}")
    else:
        lines.append("  ✓  All required fields are present.")
    lines.append("")

    lines.append("TITLE CONTINUITY")
    lines.append("─" * 50)
    if title_ok is True:
        lines.append("  ✓  Title chain is intact.")
        for note in title_notes:
            lines.append(f"     {note}")
    elif title_ok is False:
        lines.append("  ✗  Title chain continuity is BROKEN or unclear.")
        for note in title_notes:
            lines.append(f"     {note}")
    else:
        lines.append("  –  Title continuity could not be determined (missing entities).")
    lines.append("")

    lines.append("COMPLIANCE FINDINGS  (Bengaluru / Karnataka)")
    lines.append("─" * 50)
    if compliance_passed:
        lines.append("  Passed:")
        for item in compliance_passed:
            lines.append(f"    ✓  {item}")
    if compliance_failed:
        lines.append("  Failed:")
        for item in compliance_failed:
            lines.append(f"    ✗  {item}")
    if not compliance_failed and not compliance_passed:
        lines.append("  –  No compliance data available.")
    lines.append("")

    lines.append("RECOMMENDED ACTION")
    lines.append("─" * 50)
    if mandatory or risk_label == "Red":
        lines.append(
            "  MANDATORY LEGAL REVIEW: A qualified property lawyer must examine this case "
            "in full before any transaction is initiated. Do not proceed."
        )
    elif risk_label == "Yellow":
        lines.append(
            "  ADVISORY REVIEW: Legal counsel should verify the missing fields and "
            "compliance findings listed above. Re-upload corrected documents where applicable."
        )
    else:
        lines.append(
            "  PROCEED WITH STANDARD CAUTION: No blocking issues found. "
            "Standard title verification and registration checks are sufficient."
        )
    lines.append("")

    return "\n".join(lines)


def _build_llm_prompt(s: dict[str, Any]) -> str:
    """Builds the instruction prompt sent to the LLM."""
    structured = _build_structured_summary(s)
    return (
        "You are a senior property due-diligence assistant specialising in Bengaluru/Karnataka "
        "real estate law. Using the structured analysis below, write a professional legal "
        "advisory note addressed to a property lawyer. Keep the same section headings. "
        "Use formal language. Do not add any new information not present in the input. "
        "Keep each section concise.\n\n"
        f"{structured}"
    )


def _call_openrouter_chat(prompt: str) -> str:
    """
    Calls OpenRouter using an OpenAI-compatible chat completions API.
    Returns text or empty string on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return ""

    model = _preferred_openrouter_model()
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "RealProp MVP"),
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior property due-diligence assistant specialising in Bengaluru/Karnataka "
                    "real estate law. Return a concise, professional, structured advisory note."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        content = _safe_message_content(data)
        return content.strip()
    except Exception as exc:
        logger.warning("OpenRouter call failed (%s). Falling back.", exc)
        return ""


def generate_explanations_llm(case_summary_struct: dict[str, Any]) -> str:
    """
    Generate a structured, lawyer-facing due-diligence summary.

    Tries:
        1. OpenRouter if OPENROUTER_API_KEY is configured.
        2. Hugging Face transformers (flan-t5-base) if installed.
        3. Deterministic structured template fallback.

    Input:  case_summary_struct – enriched dict from run_agents()
    Output: str – structured multi-section legal advisory note
    """
    prompt = _build_llm_prompt(case_summary_struct)

    openrouter_output = _call_openrouter_chat(prompt)
    if openrouter_output and len(openrouter_output) > 100:
        return openrouter_output

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        generator = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=400,
        )
        result = generator(prompt)
        generated = result[0]["generated_text"].strip()
        if len(generated) > 100:
            return generated
    except ImportError:
        logger.info("transformers not installed — using structured template fallback.")
    except Exception as exc:
        logger.warning("LLM generation failed (%s) — using structured template fallback.", exc)

    return _build_structured_summary(case_summary_struct)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator – run_agents()
# ─────────────────────────────────────────────────────────────────────────────

def run_agents(
    case_id: str,
    entities: list[dict[str, Any]],
    ocr_confidence: float = 1.0,
    rule_outputs: list[ValidationResult] | None = None,
    risk_output: RiskOutput | None = None,
) -> AgentOutputs:
    """
    Orchestrate all 5 agents sequentially for a given case.
    """
    outputs = AgentOutputs(case_id=case_id)
    errors: list[str] = []

    try:
        outputs.extraction_validator = run_extraction_validator(entities, ocr_confidence)
    except Exception as exc:
        msg = f"ExtractionValidatorAgent failed: {exc}"
        logger.exception(msg)
        errors.append(msg)

    try:
        outputs.title_chain = run_title_chain_agent(entities)
    except Exception as exc:
        msg = f"TitleChainAgent failed: {exc}"
        logger.exception(msg)
        errors.append(msg)

    try:
        outputs.khata_analysis = run_khata_analysis_agent(entities)
    except Exception as exc:
        msg = f"KhataAnalysisAgent failed: {exc}"
        logger.exception(msg)
        errors.append(msg)

    try:
        outputs.compliance = run_compliance_agent(
            entities,
            title_chain_output=outputs.title_chain,
        )
    except Exception as exc:
        msg = f"ComplianceAgent failed: {exc}"
        logger.exception(msg)
        errors.append(msg)

    try:
        outputs.risk_synthesis = run_risk_synthesis_agent(
            rule_outputs=rule_outputs,
            risk_output=risk_output,
            extraction_validator=outputs.extraction_validator,
            title_chain=outputs.title_chain,
            compliance=outputs.compliance,
            case_id=case_id,
        )
    except Exception as exc:
        msg = f"RiskSynthesisAgent failed: {exc}"
        logger.exception(msg)
        errors.append(msg)

    try:
        synthesis = outputs.risk_synthesis
        flags: list[str] = []

        if outputs.title_chain and not outputs.title_chain.continuity_ok:
            flags.extend(outputs.title_chain.notes)

        if outputs.extraction_validator:
            flags.extend(outputs.extraction_validator.recommendations)

        compliance_passed: list[str] = []
        compliance_failed: list[str] = []
        if outputs.compliance:
            for item in outputs.compliance.checklist:
                if item.passed:
                    compliance_passed.append(item.description)
                else:
                    compliance_failed.append(f"{item.description} ({item.severity})")
                    flags.append(f"[{item.severity.upper()}] {item.description}")

        flags = _dedupe_preserve_order(flags)

        summary_struct = {
            "case_id": case_id,
            "risk_score": synthesis.risk_score if synthesis else 0,
            "risk_label": synthesis.risk_label.value if synthesis else "Unknown",
            "mandatory_review": synthesis.mandatory_review if synthesis else False,
            "flags": flags,
            "missing_fields": outputs.extraction_validator.missing_fields
            if outputs.extraction_validator
            else [],
            "title_continuity_ok": outputs.title_chain.continuity_ok
            if outputs.title_chain
            else None,
            "title_continuity_notes": outputs.title_chain.notes
            if outputs.title_chain
            else [],
            "compliance_passed": compliance_passed,
            "compliance_failed": compliance_failed,
        }

        outputs.llm_explanations = generate_explanations_llm(summary_struct)
    except Exception as exc:
        msg = f"LLM explanation generation failed: {exc}"
        logger.exception(msg)
        errors.append(msg)
        outputs.llm_explanations = _build_structured_summary(
            {
                "case_id": case_id,
                "risk_score": outputs.risk_synthesis.risk_score if outputs.risk_synthesis else 0,
                "risk_label": outputs.risk_synthesis.risk_label.value if outputs.risk_synthesis else "Unknown",
                "mandatory_review": outputs.risk_synthesis.mandatory_review if outputs.risk_synthesis else False,
                "flags": [],
                "missing_fields": outputs.extraction_validator.missing_fields
                if outputs.extraction_validator
                else [],
                "title_continuity_ok": outputs.title_chain.continuity_ok
                if outputs.title_chain
                else None,
                "title_continuity_notes": outputs.title_chain.notes
                if outputs.title_chain
                else [],
                "compliance_passed": [],
                "compliance_failed": [],
            }
        )

    outputs.errors = errors
    return outputs