# src/pipelines/validation_pipeline.py
"""
Phase 8.1 – Cross-Document Validation Pipeline

Reads extracted entity JSON files from:
    data/extracted/<case_id>/<document_id>/entities.json

Builds per-case entity views and validates:
  - Mother Deed buyer_name vs Khata owner_name
  - Property ID consistency
  - Area consistency
  - Missing mandatory seller/buyer/owner fields

Writes outputs to:
    data/results/validations/<case_id>/validation_report.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from src.core.validation_models import (
    Severity,
    ValidationType,
    ValidationResult,
)

logger = logging.getLogger("realprop.validation_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [validation_pipeline] %(message)s")

EXTRACTED_BASE_DIR = Path("data/extracted")
VALIDATION_OUT_DIR = Path("data/results/validations")

OWNER_NAME_FUZZY_THRESHOLD = 85
PROPERTY_ID_FUZZY_THRESHOLD = 90
AREA_TOLERANCE_PCT = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    noise = {"s/o", "w/o", "d/o", "mr", "mrs", "sri", "smt", "shri"}
    tokens = name.lower().split()
    tokens = [t.strip(".,") for t in tokens if t.strip(".,") not in noise]
    return " ".join(tokens).strip()


def _normalize_property_id(pid: Optional[str]) -> str:
    if not pid:
        return ""
    return pid.upper().replace(" ", "").replace("/", "").replace("-", "").strip()


def _parse_area(area_raw: Optional[str]) -> Optional[float]:
    if not area_raw:
        return None
    import re
    match = re.search(r"[\d,]+\.?\d*", str(area_raw).replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _validation_to_dict(v: ValidationResult) -> dict:
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if hasattr(v, "dict"):
        return v.dict()
    return {
        "rule_id": getattr(v, "rule_id", ""),
        "type": getattr(v, "type", ""),
        "severity": getattr(v, "severity", ""),
        "description": getattr(v, "description", ""),
        "passed": getattr(v, "passed", False),
        "evidence": getattr(v, "evidence", {}),
        "mandatory_review": getattr(v, "mandatory_review", False),
    }


def _load_entity_payloads() -> list[dict]:
    payloads: list[dict] = []
    if not EXTRACTED_BASE_DIR.exists():
        logger.warning("Extracted base dir does not exist: %s", EXTRACTED_BASE_DIR)
        return payloads

    for path in sorted(EXTRACTED_BASE_DIR.rglob("entities.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payloads.append(json.load(f))
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)

    logger.info("Loaded %d extracted entity payload(s).", len(payloads))
    return payloads


def _payload_to_field_map(payload: dict) -> dict:
    fields: dict[str, Any] = {}
    for entity in payload.get("entities", []):
        field_name = entity.get("field_name")
        value = entity.get("normalized_value") or entity.get("value")

        if not field_name:
            continue

        if field_name not in fields:
            fields[field_name] = value

    return fields


def _build_case_entities(payloads: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Builds structure:
    {
      case_id: {
        "mother_deed": {...},
        "khata": {...}
      }
    }
    """
    cases: dict[str, dict[str, dict]] = {}

    for payload in payloads:
        case_id = payload.get("case_id", "unknown_case")
        doc_type = str(payload.get("doc_type", "")).strip().lower()
        fields = _payload_to_field_map(payload)

        if case_id not in cases:
            cases[case_id] = {"mother_deed": {}, "khata": {}}

        if doc_type == "motherdeed":
            cases[case_id]["mother_deed"] = fields
        elif doc_type == "khata":
            cases[case_id]["khata"] = fields
        else:
            logger.warning("Skipping unsupported doc_type=%s in case=%s", doc_type, case_id)

    return cases


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def validate_owner_names(case_entities: Dict[str, Any]) -> ValidationResult:
    md: Dict = case_entities.get("mother_deed", {})
    kh: Dict = case_entities.get("khata", {})

    buyer_raw: Optional[str] = md.get("buyer_name") or md.get("buyer")
    owner_raw: Optional[str] = kh.get("owner_name") or kh.get("owner")

    evidence: Dict[str, Any] = {
        "mother_deed.buyer_name": buyer_raw,
        "khata.owner_name": owner_raw,
    }

    if not buyer_raw and not owner_raw:
        return ValidationResult(
            rule_id="MISSING_OWNER_NAME",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.CRITICAL,
            description="Owner/buyer name is absent in both documents.",
            passed=False,
            evidence=evidence,
            mandatory_review=True,
        )

    if not buyer_raw:
        return ValidationResult(
            rule_id="MISSING_BUYER_SELLER_NAMES",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.HIGH,
            description="Buyer name is absent in the Mother Deed.",
            passed=False,
            evidence=evidence,
            mandatory_review=True,
        )

    if not owner_raw:
        return ValidationResult(
            rule_id="MISSING_OWNER_NAME",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.CRITICAL,
            description="Owner name is absent in the Khata certificate.",
            passed=False,
            evidence=evidence,
            mandatory_review=True,
        )

    norm_buyer = _normalize_name(buyer_raw)
    norm_owner = _normalize_name(owner_raw)
    score = fuzz.token_sort_ratio(norm_buyer, norm_owner)

    evidence["normalized_buyer"] = norm_buyer
    evidence["normalized_owner"] = norm_owner
    evidence["fuzzy_score"] = score
    evidence["threshold"] = OWNER_NAME_FUZZY_THRESHOLD

    if score >= OWNER_NAME_FUZZY_THRESHOLD:
        return ValidationResult(
            rule_id="OWNER_NAME_MATCH",
            type=ValidationType.OWNER_NAME,
            severity=Severity.INFO,
            description=f"Mother Deed buyer matches Khata owner (score {score}).",
            passed=True,
            evidence=evidence,
        )

    return ValidationResult(
        rule_id="OWNER_NAME_MISMATCH",
        type=ValidationType.OWNER_NAME,
        severity=Severity.HIGH,
        description=f"Owner name mismatch (score {score} < {OWNER_NAME_FUZZY_THRESHOLD}).",
        passed=False,
        evidence=evidence,
    )


def validate_property_ids(case_entities: Dict[str, Any]) -> ValidationResult:
    md: Dict = case_entities.get("mother_deed", {})
    kh: Dict = case_entities.get("khata", {})

    md_pid_raw: Optional[str] = (
        md.get("property_id")
        or md.get("survey_number")
        or md.get("site_number")
        or md.get("flat_number")
    )
    kh_pid_raw: Optional[str] = (
        kh.get("property_id")
        or kh.get("khata_property_id")
        or kh.get("site_number")
    )

    evidence: Dict[str, Any] = {
        "mother_deed.property_id": md_pid_raw,
        "khata.property_id": kh_pid_raw,
    }

    if not md_pid_raw and not kh_pid_raw:
        return ValidationResult(
            rule_id="MISSING_PROPERTY_ID",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.HIGH,
            description="Property ID is absent in both documents.",
            passed=False,
            evidence=evidence,
        )

    if not md_pid_raw:
        return ValidationResult(
            rule_id="MISSING_PROPERTY_ID",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.HIGH,
            description="Property ID is absent in the Mother Deed.",
            passed=False,
            evidence=evidence,
        )

    if not kh_pid_raw:
        return ValidationResult(
            rule_id="MISSING_PROPERTY_ID",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.HIGH,
            description="Property ID is absent in the Khata certificate.",
            passed=False,
            evidence=evidence,
        )

    norm_md = _normalize_property_id(md_pid_raw)
    norm_kh = _normalize_property_id(kh_pid_raw)
    score = fuzz.ratio(norm_md, norm_kh)

    evidence["normalized_md_id"] = norm_md
    evidence["normalized_kh_id"] = norm_kh
    evidence["fuzzy_score"] = score
    evidence["threshold"] = PROPERTY_ID_FUZZY_THRESHOLD

    if score >= PROPERTY_ID_FUZZY_THRESHOLD:
        return ValidationResult(
            rule_id="PROPERTY_ID_MATCH",
            type=ValidationType.PROPERTY_ID,
            severity=Severity.INFO,
            description=f"Property IDs match (score {score}).",
            passed=True,
            evidence=evidence,
        )

    return ValidationResult(
        rule_id="PROPERTY_ID_MISMATCH",
        type=ValidationType.PROPERTY_ID,
        severity=Severity.HIGH,
        description=f"Property ID mismatch (score {score} < {PROPERTY_ID_FUZZY_THRESHOLD}).",
        passed=False,
        evidence=evidence,
    )


def validate_area(case_entities: Dict[str, Any]) -> ValidationResult:
    md: Dict = case_entities.get("mother_deed", {})
    kh: Dict = case_entities.get("khata", {})

    md_area_raw: Optional[str] = md.get("area") or md.get("plot_area") or md.get("flat_area")
    kh_area_raw: Optional[str] = kh.get("area") or kh.get("plot_area")

    evidence: Dict[str, Any] = {
        "mother_deed.area": md_area_raw,
        "khata.area": kh_area_raw,
    }

    if not md_area_raw and not kh_area_raw:
        return ValidationResult(
            rule_id="MISSING_AREA",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.LOW,
            description="Area field is absent in both documents; area check skipped.",
            passed=False,
            evidence=evidence,
        )

    if not md_area_raw or not kh_area_raw:
        missing_in = "Mother Deed" if not md_area_raw else "Khata certificate"
        return ValidationResult(
            rule_id="MISSING_AREA",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.LOW,
            description=f"Area field is absent in the {missing_in}; area check skipped.",
            passed=False,
            evidence=evidence,
        )

    md_area = _parse_area(md_area_raw)
    kh_area = _parse_area(kh_area_raw)

    if md_area is None or kh_area is None or kh_area == 0:
        return ValidationResult(
            rule_id="MISSING_AREA",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.LOW,
            description="Could not parse numeric area from one or both documents.",
            passed=False,
            evidence=evidence,
        )

    diff_pct = abs(md_area - kh_area) / kh_area * 100
    evidence["md_area_parsed"] = md_area
    evidence["kh_area_parsed"] = kh_area
    evidence["diff_pct"] = round(diff_pct, 2)
    evidence["tolerance_pct"] = AREA_TOLERANCE_PCT

    if diff_pct <= AREA_TOLERANCE_PCT:
        return ValidationResult(
            rule_id="AREA_MATCH",
            type=ValidationType.AREA,
            severity=Severity.INFO,
            description=f"Area values are consistent (diff {diff_pct:.1f}%).",
            passed=True,
            evidence=evidence,
        )

    return ValidationResult(
        rule_id="AREA_MISMATCH",
        type=ValidationType.AREA,
        severity=Severity.MEDIUM,
        description=f"Area mismatch (diff {diff_pct:.1f}% > tolerance {AREA_TOLERANCE_PCT}%).",
        passed=False,
        evidence=evidence,
    )


def validate_missing_seller_name(case_entities: Dict[str, Any]) -> ValidationResult:
    md: Dict = case_entities.get("mother_deed", {})
    seller_raw: Optional[str] = md.get("seller_name") or md.get("seller")
    evidence: Dict[str, Any] = {"mother_deed.seller_name": seller_raw}

    if not seller_raw:
        return ValidationResult(
            rule_id="MISSING_BUYER_SELLER_NAMES",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.HIGH,
            description="Seller name is absent in the Mother Deed.",
            passed=False,
            evidence=evidence,
            mandatory_review=True,
        )

    return ValidationResult(
        rule_id="SELLER_NAME_PRESENT",
        type=ValidationType.MISSING_FIELD,
        severity=Severity.INFO,
        description=f"Seller name present in Mother Deed: '{seller_raw}'.",
        passed=True,
        evidence=evidence,
    )


def collect_validation_results(case_id: str, case_entities: Dict[str, Any]) -> List[ValidationResult]:
    logger.info("=== Starting validation for case_id=%s ===", case_id)

    results: List[ValidationResult] = [
        validate_owner_names(case_entities),
        validate_missing_seller_name(case_entities),
        validate_property_ids(case_entities),
        validate_area(case_entities),
    ]

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    logger.info("Validation complete for case_id=%s: %d passed, %d failed.", case_id, passed, failed)

    return results


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_validation_pipeline() -> None:
    VALIDATION_OUT_DIR.mkdir(parents=True, exist_ok=True)

    payloads = _load_entity_payloads()
    if not payloads:
        summary_path = VALIDATION_OUT_DIR / "validation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "no_extracted_payloads_found",
                    "base_dir": str(EXTRACTED_BASE_DIR),
                    "case_count": 0,
                    "reports": [],
                },
                f,
                indent=2,
            )
        logger.warning("No extracted payloads found. Wrote empty summary → %s", summary_path)
        return

    case_map = _build_case_entities(payloads)
    reports = []

    for case_id, case_entities in case_map.items():
        results = collect_validation_results(case_id, case_entities)
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        mandatory_review = any(bool(getattr(r, "mandatory_review", False)) for r in results)

        case_dir = VALIDATION_OUT_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "case_id": case_id,
            "generated_from": str(EXTRACTED_BASE_DIR),
            "passed_checks": passed,
            "failed_checks": failed,
            "mandatory_review": mandatory_review,
            "case_entities": case_entities,
            "results": [_validation_to_dict(r) for r in results],
        }

        report_path = case_dir / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Saved validation report → %s", report_path)
        reports.append({
            "case_id": case_id,
            "report_path": str(report_path),
            "passed_checks": passed,
            "failed_checks": failed,
            "mandatory_review": mandatory_review,
        })

    summary_path = VALIDATION_OUT_DIR / "validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "ok",
                "generated_from": str(EXTRACTED_BASE_DIR),
                "case_count": len(reports),
                "reports": reports,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Validation pipeline complete. Summary → %s", summary_path)


if __name__ == "__main__":
    run_validation_pipeline()