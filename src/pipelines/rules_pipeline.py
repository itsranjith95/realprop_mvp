from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.validation_models import (
    RiskLabel,
    RiskOutput,
    RuleHit,
    Severity,
    ValidationResult,
)
from src.services.rules_service import evaluate_rules as evaluate_rules_service
from src.services.evidence_service import store_evidence
from src.services.prompt_service import generate_flag_explanation

logger = logging.getLogger("realprop.rules_pipeline")

_BASE_DIR = Path(__file__).resolve().parents[2]
_VALIDATIONS_DIR = _BASE_DIR / "data" / "results" / "validations"
_RISK_OUTPUT_DIR = _BASE_DIR / "data" / "results" / "risk_scores"


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
        )


def _load_validation_reports(validations_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = validations_dir or _VALIDATIONS_DIR
    if not root.exists():
        logger.warning("Validation directory not found: %s", root)
        return []

    reports: List[Dict[str, Any]] = []
    for report_path in sorted(root.glob("*/validation_report.json")):
        try:
            with open(report_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload["_report_path"] = str(report_path)
            reports.append(payload)
        except Exception:
            logger.exception("Failed to load validation report: %s", report_path)

    logger.info("Loaded %d validation report(s).", len(reports))
    return reports


def _parse_validation_results(report_payload: Dict[str, Any]) -> List[ValidationResult]:
    raw_results = report_payload.get("results", [])
    parsed: List[ValidationResult] = []

    for item in raw_results:
        try:
            parsed.append(ValidationResult.model_validate(item))
        except Exception:
            logger.exception(
                "Failed to parse ValidationResult for case_id=%s item=%s",
                report_payload.get("case_id"),
                item,
            )

    return parsed


def _map_service_label_to_risk_label(label_text: str) -> RiskLabel:
    normalized = (label_text or "").lower()
    if "high" in normalized or "red" in normalized:
        return RiskLabel.RED
    if "medium" in normalized or "yellow" in normalized:
        return RiskLabel.YELLOW
    return RiskLabel.GREEN


def _convert_service_hits_to_rule_hits(service_result: Dict[str, Any]) -> List[RuleHit]:
    rule_hits: List[RuleHit] = []

    for item in service_result.get("rule_hits", []):
        severity_value = str(item.get("severity", "medium")).lower()
        mandatory_review = bool(item.get("requires_review", False))
        evidence = item.get("evidence", {}) or {}

        hit = RuleHit(
            rule_id=item.get("rule_id", ""),
            name=item.get("rule_name", item.get("rule_id", "")),
            severity=Severity(severity_value),
            points=int(item.get("points", 0)),
            mandatory_review=mandatory_review,
            evidence=evidence,
        )
        rule_hits.append(hit)

    return rule_hits


def _store_rule_hit_evidence(
    case_id: str,
    case_entities: Dict[str, Any],
    rule_hits: List[RuleHit],
) -> None:
    """
    Store lightweight evidence records for each triggered rule hit.
    Tries to extract document/page/bbox from rule evidence payload when present.
    """
    for hit in rule_hits:
        evidence = hit.evidence or {}

        source_document_id = (
            evidence.get("source_document_id")
            or evidence.get("document_id")
            or evidence.get("doc_id")
            or "unknown_document"
        )
        page = evidence.get("page", 0)
        bbox = evidence.get("bbox", "")
        field_name = evidence.get("field") or evidence.get("field_name") or ""
        extracted_value = (
            evidence.get("value")
            or evidence.get("left_value")
            or evidence.get("right_value")
            or ""
        )
        doc_type = evidence.get("doc_type", "")
        rule_version = evidence.get("rule_version", "unknown")

        try:
            store_evidence(
                case_id=case_id,
                source_document_id=source_document_id,
                rule_id=hit.rule_id,
                rule_version=rule_version,
                page=page if isinstance(page, int) else 0,
                bbox=bbox if isinstance(bbox, list) else None,
                field_name=field_name,
                extracted_value=str(extracted_value),
                doc_type=doc_type,
                rule_name=hit.name,
                severity=hit.severity.value,
                flag_description=f"{hit.name} triggered in rules pipeline",
            )
        except Exception:
            logger.exception(
                "Failed to store evidence for case_id=%s rule_id=%s",
                case_id,
                hit.rule_id,
            )


def _attach_llm_explanations(rule_hits: List[RuleHit], prefer_llm: str = "openrouter") -> List[RuleHit]:
    """
    Optionally add plain-English explanation to each rule hit evidence.
    Safe fallback: if LLM fails, pipeline continues.
    """
    enriched_hits: List[RuleHit] = []

    for hit in rule_hits:
        evidence = hit.evidence or {}
        try:
            explanation = generate_flag_explanation(
                rule_id=hit.rule_id,
                rule_name=hit.name,
                rule_version=evidence.get("rule_version", "unknown"),
                severity=hit.severity.value,
                rule_description=f"{hit.name} detected during due-diligence review",
                evidence_summary=json.dumps(evidence, default=str)[:1000],
                source_document_id=evidence.get("source_document_id", "unknown_document"),
                page=evidence.get("page", 0),
                bbox=str(evidence.get("bbox", "")),
                extracted_values=json.dumps(evidence, default=str)[:1000],
                prefer_llm=prefer_llm,
            )
            updated_evidence = dict(evidence)
            updated_evidence["llm_explanation"] = explanation
            enriched_hits.append(
                RuleHit(
                    rule_id=hit.rule_id,
                    name=hit.name,
                    severity=hit.severity,
                    points=hit.points,
                    mandatory_review=hit.mandatory_review,
                    evidence=updated_evidence,
                )
            )
        except Exception:
            logger.exception("Failed to generate explanation for rule_id=%s", hit.rule_id)
            enriched_hits.append(hit)

    return enriched_hits


def run_rules_pipeline(
    case_id: str,
    case_entities: Dict[str, Any],
    validations: List[ValidationResult],
    persist: bool = True,
    output_dir: Optional[Path] = None,
    attach_explanations: bool = False,
) -> RiskOutput:
    """
    Orchestrates rules evaluation using the service layer,
    stores evidence, optionally generates explanations, and persists RiskOutput.
    """
    service_result = evaluate_rules_service(case_id=case_id, entities=case_entities)

    score = int(service_result.get("risk_score", 0))
    label = _map_service_label_to_risk_label(service_result.get("risk_label", "Low Risk"))
    rule_hits = _convert_service_hits_to_rule_hits(service_result)

    if attach_explanations:
        rule_hits = _attach_llm_explanations(rule_hits)

    _store_rule_hit_evidence(case_id, case_entities, rule_hits)

    mandatory_review = any(h.mandatory_review for h in rule_hits)

    failed = [v for v in validations if not v.passed]
    summary_parts = [f"Score: {score} ({label.value})"]
    if mandatory_review:
        summary_parts.append("Mandatory review required")
    if failed:
        summary_parts.append(f"Issues: {', '.join(v.rule_id for v in failed)}")
    else:
        if rule_hits:
            summary_parts.append(f"Issues: {', '.join(h.rule_id for h in rule_hits)}")

    risk_output = RiskOutput(
        case_id=case_id,
        risk_score=score,
        risk_label=label,
        mandatory_review=mandatory_review,
        rule_hits=rule_hits,
        validation_results=validations,
        summary=" | ".join(summary_parts),
    )

    if persist:
        persist_risk_output(risk_output, output_dir=output_dir)

    return risk_output


def persist_risk_output(
    risk_output: RiskOutput,
    output_dir: Optional[Path] = None,
) -> Path:
    out_dir = output_dir or _RISK_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{risk_output.case_id}_risk.json"
    payload = risk_output.model_dump()
    payload["risk_label"] = risk_output.risk_label.value
    payload["persisted_at"] = datetime.now(timezone.utc).isoformat()

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Risk output persisted → %s", out_path)
    return out_path


def persist_risk_summary(
    risk_outputs: List[RiskOutput],
    output_dir: Optional[Path] = None,
) -> Path:
    out_dir = output_dir or _RISK_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(risk_outputs),
        "cases": [
            {
                "case_id": item.case_id,
                "risk_score": item.risk_score,
                "risk_label": item.risk_label.value,
                "mandatory_review": item.mandatory_review,
                "rule_hit_count": len(item.rule_hits),
            }
            for item in risk_outputs
        ],
    }

    out_path = out_dir / "risk_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Risk summary persisted → %s", out_path)
    return out_path


def main() -> None:
    _setup_logging()

    reports = _load_validation_reports()
    if not reports:
        _RISK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        empty_summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_cases": 0,
            "cases": [],
        }
        out_path = _RISK_OUTPUT_DIR / "risk_summary.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(empty_summary, fh, indent=2)
        logger.warning("No validation reports found. Wrote empty summary → %s", out_path)
        return

    all_outputs: List[RiskOutput] = []

    for report in reports:
        case_id = report.get("case_id")
        if not case_id:
            logger.warning("Skipping validation report with missing case_id: %s", report.get("_report_path"))
            continue

        case_entities = report.get("case_entities", {}) or {}
        validations = _parse_validation_results(report)

        risk_output = run_rules_pipeline(
            case_id=case_id,
            case_entities=case_entities,
            validations=validations,
            persist=True,
            output_dir=_RISK_OUTPUT_DIR,
            attach_explanations=False,
        )
        all_outputs.append(risk_output)

    persist_risk_summary(all_outputs, output_dir=_RISK_OUTPUT_DIR)
    logger.info("Rules pipeline complete. Processed %d case(s).", len(all_outputs))


if __name__ == "__main__":
    main()