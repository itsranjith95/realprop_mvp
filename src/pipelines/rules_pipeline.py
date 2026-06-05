from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.core.validation_models import (
    RiskLabel,
    RiskOutput,
    RuleHit,
    Severity,
    ValidationResult,
)

logger = logging.getLogger("realprop.rules_pipeline")

_BASE_DIR = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _BASE_DIR / "config" / "rules_config.yaml"
_VALIDATIONS_DIR = _BASE_DIR / "data" / "results" / "validations"
_RISK_OUTPUT_DIR = _BASE_DIR / "data" / "results" / "risk_scores"


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
        )


def _load_rules_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Rules config not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    logger.debug("Loaded rules config from %s (version=%s)", path, config.get("version"))
    return config


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


def evaluate_rules(
    case_id: str,
    validations: List[ValidationResult],
    entities: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> Tuple[int, RiskLabel, List[RuleHit]]:
    logger.info("=== Evaluating rules for case_id=%s ===", case_id)

    config = _load_rules_config(config_path)
    rules_index: Dict[str, Dict[str, Any]] = {r["id"]: r for r in config.get("rules", [])}
    confidence_cfg = config.get("confidence_thresholds", {})
    min_confidence: float = float(confidence_cfg.get("critical_field_min", 0.70))
    critical_fields: List[str] = confidence_cfg.get("critical_fields", [])
    thresholds = config.get("risk_thresholds", {})

    failed_rules: Dict[str, ValidationResult] = {
        v.rule_id: v for v in validations if not v.passed
    }

    total_points = 0
    rule_hits: List[RuleHit] = []
    triggered_ids = set()

    for rule_id, validation in failed_rules.items():
        if rule_id not in rules_index:
            logger.warning(
                "Validation rule_id '%s' not found in rules_config; skipping scoring.",
                rule_id,
            )
            continue

        rule_cfg = rules_index[rule_id]
        points = int(rule_cfg.get("points", 0))
        total_points += points
        triggered_ids.add(rule_id)

        hit = RuleHit(
            rule_id=rule_id,
            name=rule_cfg.get("name", rule_id),
            severity=Severity(rule_cfg.get("severity", "medium")),
            points=points,
            mandatory_review=bool(rule_cfg.get("mandatory_review", False)),
            evidence=validation.evidence or {},
        )
        rule_hits.append(hit)

        logger.info(
            "Rule HIT: %s (%s) +%d points",
            rule_id,
            rule_cfg.get("severity"),
            points,
        )

    if "LOW_CONFIDENCE_CRITICAL_FIELD" not in triggered_ids:
        low_conf_evidence: Dict[str, Any] = {}

        for doc_key in ("mother_deed", "khata"):
            doc_conf = entities.get(doc_key, {}).get("extraction_confidence", {}) or {}
            for field in critical_fields:
                conf_val = doc_conf.get(field)
                if conf_val is None:
                    continue
                try:
                    conf_float = float(conf_val)
                except (TypeError, ValueError):
                    continue
                if conf_float < min_confidence:
                    low_conf_evidence[f"{doc_key}.{field}"] = conf_float

        if low_conf_evidence:
            rule_cfg = rules_index.get("LOW_CONFIDENCE_CRITICAL_FIELD", {})
            points = int(rule_cfg.get("points", 15))
            total_points += points

            hit = RuleHit(
                rule_id="LOW_CONFIDENCE_CRITICAL_FIELD",
                name=rule_cfg.get("name", "Low Confidence on Critical Field"),
                severity=Severity(rule_cfg.get("severity", "medium")),
                points=points,
                mandatory_review=bool(rule_cfg.get("mandatory_review", False)),
                evidence=low_conf_evidence,
            )
            rule_hits.append(hit)

            logger.info(
                "Rule HIT: LOW_CONFIDENCE_CRITICAL_FIELD +%d points | fields=%s",
                points,
                list(low_conf_evidence.keys()),
            )

    risk_score = min(total_points, 100)

    risk_label = RiskLabel.GREEN
    for label_key in ("red", "yellow", "green"):
        band = thresholds.get(label_key, {})
        if band.get("min", 0) <= risk_score <= band.get("max", 0):
            risk_label = RiskLabel(band["label"])
            break

    logger.info(
        "Risk evaluation complete for case_id=%s: score=%d label=%s hits=%d",
        case_id,
        risk_score,
        risk_label.value,
        len(rule_hits),
    )
    return risk_score, risk_label, rule_hits


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


def run_rules_pipeline(
    case_id: str,
    case_entities: Dict[str, Any],
    validations: List[ValidationResult],
    persist: bool = True,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> RiskOutput:
    score, label, rule_hits = evaluate_rules(
        case_id=case_id,
        validations=validations,
        entities=case_entities,
        config_path=config_path,
    )

    mandatory_review = any(h.mandatory_review for h in rule_hits)

    failed = [v for v in validations if not v.passed]
    summary_parts = [f"Score: {score} ({label.value})"]
    if mandatory_review:
        summary_parts.append("Mandatory review required")
    if failed:
        summary_parts.append(f"Issues: {', '.join(v.rule_id for v in failed)}")

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
            config_path=_CONFIG_PATH,
        )
        all_outputs.append(risk_output)

    persist_risk_summary(all_outputs, output_dir=_RISK_OUTPUT_DIR)
    logger.info("Rules pipeline complete. Processed %d case(s).", len(all_outputs))


if __name__ == "__main__":
    main()