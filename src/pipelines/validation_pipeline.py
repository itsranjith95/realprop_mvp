# src/pipelines/validation_pipeline.py
"""
Phase 5 – Cross-Document Validation Pipeline
Validates Mother Deed ↔ Khata entities for:
  - Owner / buyer name consistency
  - Property ID consistency
  - Area consistency
  - Missing mandatory fields
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from src.core.validation_models import (
    Severity,
    ValidationType,
    ValidationResult,
)

logger = logging.getLogger("realprop.validation_pipeline")

# ---------------------------------------------------------------------------
# Constants (overridable via config)
# ---------------------------------------------------------------------------
OWNER_NAME_FUZZY_THRESHOLD = 85      # rapidfuzz token_sort_ratio minimum
PROPERTY_ID_FUZZY_THRESHOLD = 90
AREA_TOLERANCE_PCT = 5.0             # percent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: Optional[str]) -> str:
    """Lowercase, strip extra spaces, remove common noise words."""
    if not name:
        return ""
    noise = {"s/o", "w/o", "d/o", "mr", "mrs", "sri", "smt", "shri"}
    tokens = name.lower().split()
    tokens = [t.strip(".,") for t in tokens if t.strip(".,") not in noise]
    return " ".join(tokens).strip()


def _normalize_property_id(pid: Optional[str]) -> str:
    """Uppercase, remove spaces/slashes for comparison."""
    if not pid:
        return ""
    return pid.upper().replace(" ", "").replace("/", "").replace("-", "").strip()


def _parse_area(area_raw: Optional[str]) -> Optional[float]:
    """Extract numeric area value from a raw string like '1200 sqft' or '120 sq.mt'."""
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


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def validate_owner_names(case_entities: Dict[str, Any]) -> ValidationResult:
    """
    Compare Mother Deed buyer_name vs Khata owner_name using fuzzy matching.

    Parameters
    ----------
    case_entities : dict with keys 'mother_deed' and 'khata', each a dict
                    of extracted entity fields.

    Returns
    -------
    ValidationResult
    """
    md: Dict = case_entities.get("mother_deed", {})
    kh: Dict = case_entities.get("khata", {})

    buyer_raw: Optional[str] = md.get("buyer_name") or md.get("buyer")
    owner_raw: Optional[str] = kh.get("owner_name") or kh.get("owner")

    evidence: Dict[str, Any] = {
        "mother_deed.buyer_name": buyer_raw,
        "khata.owner_name": owner_raw,
    }

    # --- Missing field checks ---
    if not buyer_raw and not owner_raw:
        logger.warning("Both buyer_name and owner_name are missing for case.")
        return ValidationResult(
            rule_id="MISSING_OWNER_NAME",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.CRITICAL,
            description=(
                "Owner/buyer name is absent in BOTH documents. "
                "Ownership validation is impossible."
            ),
            passed=False,
            evidence=evidence,
            mandatory_review=True,
        )

    if not buyer_raw:
        logger.warning("Mother Deed buyer_name is missing.")
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
        logger.warning("Khata owner_name is missing.")
        return ValidationResult(
            rule_id="MISSING_OWNER_NAME",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.CRITICAL,
            description="Owner name is absent in the Khata certificate.",
            passed=False,
            evidence=evidence,
            mandatory_review=True,
        )

    # --- Fuzzy match ---
    norm_buyer = _normalize_name(buyer_raw)
    norm_owner = _normalize_name(owner_raw)
    score = fuzz.token_sort_ratio(norm_buyer, norm_owner)

    evidence["normalized_buyer"] = norm_buyer
    evidence["normalized_owner"] = norm_owner
    evidence["fuzzy_score"] = score
    evidence["threshold"] = OWNER_NAME_FUZZY_THRESHOLD

    if score >= OWNER_NAME_FUZZY_THRESHOLD:
        logger.info(
            "Owner name check PASSED: '%s' ↔ '%s' (score=%d)",
            norm_buyer, norm_owner, score,
        )
        return ValidationResult(
            rule_id="OWNER_NAME_MATCH",
            type=ValidationType.OWNER_NAME,
            severity=Severity.INFO,
            description=(
                f"Mother Deed buyer '{buyer_raw}' matches Khata owner "
                f"'{owner_raw}' (fuzzy score {score})."
            ),
            passed=True,
            evidence=evidence,
        )
    else:
        logger.warning(
            "Owner name MISMATCH: '%s' ↔ '%s' (score=%d < %d)",
            norm_buyer, norm_owner, score, OWNER_NAME_FUZZY_THRESHOLD,
        )
        return ValidationResult(
            rule_id="OWNER_NAME_MISMATCH",
            type=ValidationType.OWNER_NAME,
            severity=Severity.HIGH,
            description=(
                f"Owner name mismatch: Mother Deed buyer='{buyer_raw}' vs "
                f"Khata owner='{owner_raw}' (fuzzy score {score} < "
                f"{OWNER_NAME_FUZZY_THRESHOLD})."
            ),
            passed=False,
            evidence=evidence,
        )


def validate_property_ids(case_entities: Dict[str, Any]) -> ValidationResult:
    """
    Compare Mother Deed property_id vs Khata property_id.

    Parameters
    ----------
    case_entities : dict with keys 'mother_deed' and 'khata'.

    Returns
    -------
    ValidationResult
    """
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
        logger.info(
            "Property ID check PASSED: '%s' ↔ '%s' (score=%d)",
            norm_md, norm_kh, score,
        )
        return ValidationResult(
            rule_id="PROPERTY_ID_MATCH",
            type=ValidationType.PROPERTY_ID,
            severity=Severity.INFO,
            description=(
                f"Property IDs match: '{md_pid_raw}' ↔ '{kh_pid_raw}' "
                f"(score {score})."
            ),
            passed=True,
            evidence=evidence,
        )
    else:
        logger.warning(
            "Property ID MISMATCH: '%s' ↔ '%s' (score=%d < %d)",
            norm_md, norm_kh, score, PROPERTY_ID_FUZZY_THRESHOLD,
        )
        return ValidationResult(
            rule_id="PROPERTY_ID_MISMATCH",
            type=ValidationType.PROPERTY_ID,
            severity=Severity.HIGH,
            description=(
                f"Property ID mismatch: Mother Deed='{md_pid_raw}' vs "
                f"Khata='{kh_pid_raw}' (score {score} < "
                f"{PROPERTY_ID_FUZZY_THRESHOLD})."
            ),
            passed=False,
            evidence=evidence,
        )


def validate_area(case_entities: Dict[str, Any]) -> ValidationResult:
    """
    Compare area fields between Mother Deed and Khata (within tolerance %).

    Parameters
    ----------
    case_entities : dict with keys 'mother_deed' and 'khata'.

    Returns
    -------
    ValidationResult
    """
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
            passed=True,  # non-blocking
            evidence=evidence,
        )

    if not md_area_raw or not kh_area_raw:
        missing_in = "Mother Deed" if not md_area_raw else "Khata"
        return ValidationResult(
            rule_id="MISSING_AREA",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.LOW,
            description=f"Area field is absent in the {missing_in}; area check skipped.",
            passed=True,  # partial data → non-blocking but flagged
            evidence=evidence,
        )

    md_area = _parse_area(md_area_raw)
    kh_area = _parse_area(kh_area_raw)

    if md_area is None or kh_area is None:
        return ValidationResult(
            rule_id="MISSING_AREA",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.LOW,
            description=(
                f"Could not parse numeric area from one or both documents "
                f"(MD='{md_area_raw}', Khata='{kh_area_raw}')."
            ),
            passed=True,
            evidence=evidence,
        )

    if kh_area == 0:
        return ValidationResult(
            rule_id="MISSING_AREA",
            type=ValidationType.MISSING_FIELD,
            severity=Severity.LOW,
            description="Khata area value parsed as zero; skipping percentage check.",
            passed=True,
            evidence=evidence,
        )

    diff_pct = abs(md_area - kh_area) / kh_area * 100
    evidence["md_area_parsed"] = md_area
    evidence["kh_area_parsed"] = kh_area
    evidence["diff_pct"] = round(diff_pct, 2)
    evidence["tolerance_pct"] = AREA_TOLERANCE_PCT

    if diff_pct <= AREA_TOLERANCE_PCT:
        logger.info(
            "Area check PASSED: MD=%.2f, Khata=%.2f (diff=%.2f%%)",
            md_area, kh_area, diff_pct,
        )
        return ValidationResult(
            rule_id="AREA_MATCH",
            type=ValidationType.AREA,
            severity=Severity.INFO,
            description=(
                f"Area values are consistent: Mother Deed={md_area}, "
                f"Khata={kh_area} (diff {diff_pct:.1f}%)."
            ),
            passed=True,
            evidence=evidence,
        )
    else:
        logger.warning(
            "Area MISMATCH: MD=%.2f, Khata=%.2f (diff=%.2f%% > %.1f%%)",
            md_area, kh_area, diff_pct, AREA_TOLERANCE_PCT,
        )
        return ValidationResult(
            rule_id="AREA_MISMATCH",
            type=ValidationType.AREA,
            severity=Severity.MEDIUM,
            description=(
                f"Area mismatch: Mother Deed={md_area} vs Khata={kh_area} "
                f"(diff {diff_pct:.1f}% > tolerance {AREA_TOLERANCE_PCT}%)."
            ),
            passed=False,
            evidence=evidence,
        )


def validate_missing_seller_name(case_entities: Dict[str, Any]) -> ValidationResult:
    """Check that seller_name is present in the Mother Deed."""
    md: Dict = case_entities.get("mother_deed", {})
    seller_raw: Optional[str] = md.get("seller_name") or md.get("seller")
    evidence: Dict[str, Any] = {"mother_deed.seller_name": seller_raw}

    if not seller_raw:
        logger.warning("Seller name missing in Mother Deed.")
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def collect_validation_results(
    case_id: str,
    case_entities: Dict[str, Any],
) -> List[ValidationResult]:
    """
    Run all cross-document validation checks for a given case.

    Parameters
    ----------
    case_id      : unique case identifier (for logging)
    case_entities: dict with 'mother_deed' and 'khata' entity sub-dicts

    Returns
    -------
    List[ValidationResult] – one entry per check, pass or fail.
    """
    logger.info("=== Starting validation for case_id=%s ===", case_id)

    results: List[ValidationResult] = []

    results.append(validate_owner_names(case_entities))
    results.append(validate_missing_seller_name(case_entities))
    results.append(validate_property_ids(case_entities))
    results.append(validate_area(case_entities))

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    logger.info(
        "Validation complete for case_id=%s: %d passed, %d failed.",
        case_id, passed, failed,
    )

    return results