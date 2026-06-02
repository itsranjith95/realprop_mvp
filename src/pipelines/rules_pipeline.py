# src/pipelines/rules_pipeline.py
"""
Phase 5 – Rules & Risk Scoring Pipeline
Loads rules from config/rules_config.yaml and evaluates them against
the validation results + entity extraction confidences to produce a
deterministic 0–100 risk score with a Green / Yellow / Red label.
"""

from __future__ import annotations

import json
import logging
import os
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

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rules_config.yaml"


def _load_rules_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and parse rules_config.yaml."""
    path = config_path or _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Rules config not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    logger.debug("Loaded rules config from %s (version=%s)", path, config.get("version"))
    return config


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_rules(
    case_id: str,
    validations: List[ValidationResult],
    entities: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> Tuple[int, RiskLabel, List[RuleHit]]:
    """
    Evaluate governance rules against validation results and entity confidences.

    Parameters
    ----------
    case_id     : unique case identifier
    validations : output of collect_validation_results()
    entities    : raw entity dict (same as case_entities in validation pipeline),
                  may include extraction_confidence sub-dicts
    config_path : optional override for rules_config.yaml path

    Returns
    -------
    (risk_score, risk_label, rule_hits)
        risk_score  : int 0–100 (capped at 100)
        risk_label  : RiskLabel.GREEN / YELLOW / RED
        rule_hits   : list of RuleHit objects for triggered rules
    """
    logger.info("=== Evaluating rules for case_id=%s ===", case_id)

    config = _load_rules_config(config_path)
    rules_index: Dict[str, Dict] = {r["id"]: r for r in config.get("rules", [])}
    confidence_cfg = config.get("confidence_thresholds", {})
    min_confidence: float = confidence_cfg.get("critical_field_min", 0.70)
    critical_fields: List[str] = confidence_cfg.get("critical_fields", [])
    thresholds = config.get("risk_thresholds", {})

    # Build a quick lookup: rule_id → failed ValidationResult
    failed_rules: Dict[str, ValidationResult] = {
        v.rule_id: v for v in validations if not v.passed
    }

    total_points = 0
    rule_hits: List[RuleHit] = []
    triggered_ids: set = set()

    # --- Evaluate each failed validation against known rules ---
    for rule_id, validation in failed_rules.items():
        if rule_id not in rules_index:
            # Unknown rule_id – still record it with 0 points
            logger.debug("Validation rule_id '%s' not in rules_config; skipping scoring.", rule_id)
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
            evidence=validation.evidence,
        )
        rule_hits.append(hit)
        logger.info(
            "Rule HIT: %s (%s) +%d points | evidence=%s",
            rule_id, rule_cfg.get("severity"), points, validation.evidence,
        )

    # --- LOW_CONFIDENCE_CRITICAL_FIELD check ---
    if "LOW_CONFIDENCE_CRITICAL_FIELD" not in triggered_ids:
        low_conf_evidence: Dict[str, Any] = {}
        confidences: Dict[str, Any] = entities.get("extraction_confidence", {})

        # Check both mother_deed and khata sub-dicts
        for doc_key in ("mother_deed", "khata"):
            doc_conf = entities.get(doc_key, {}).get("extraction_confidence", {})
            for field in critical_fields:
                conf_val = doc_conf.get(field) or confidences.get(field)
                if conf_val is not None and float(conf_val) < min_confidence:
                    low_conf_evidence[f"{doc_key}.{field}"] = conf_val

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
                points, list(low_conf_evidence.keys()),
            )

    # --- Cap score at 100 ---
    risk_score = min(total_points, 100)

    # --- Map score to label ---
    risk_label = RiskLabel.GREEN
    for label_key in ("red", "yellow", "green"):
        band = thresholds.get(label_key, {})
        if band.get("min", 0) <= risk_score <= band.get("max", 0):
            risk_label = RiskLabel(band["label"])
            break

    mandatory_review = any(h.mandatory_review for h in rule_hits)

    logger.info(
        "Risk evaluation complete for case_id=%s: score=%d label=%s mandatory_review=%s hits=%d",
        case_id, risk_score, risk_label.value, mandatory_review, len(rule_hits),
    )
    return risk_score, risk_label, rule_hits


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_risk_score(
    case_id: str,
    score: int,
    label: RiskLabel,
    rule_hits: Optional[List[RuleHit]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Persist the risk score, label, and rule hits for a case to a JSON file.

    The file is written to:
        data/results/risk_scores/{case_id}_risk.json

    Parameters
    ----------
    case_id    : unique case identifier
    score      : risk score 0–100
    label      : RiskLabel enum value
    rule_hits  : list of triggered RuleHit objects (optional)
    output_dir : override output directory (defaults to data/results/risk_scores)

    Returns
    -------
    Path to the written file.
    """
    if output_dir is None:
        base = Path(__file__).resolve().parents[2]
        output_dir = base / "data" / "results" / "risk_scores"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}_risk.json"

    payload = {
        "case_id": case_id,
        "risk_score": score,
        "risk_label": label.value,
        "mandatory_review": any(h.mandatory_review for h in (rule_hits or [])),
        "rule_hits": [h.model_dump() for h in (rule_hits or [])],
        "persisted_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Risk score persisted → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: full end-to-end runner for a case
# ---------------------------------------------------------------------------

def run_rules_pipeline(
    case_id: str,
    case_entities: Dict[str, Any],
    validations: List[ValidationResult],
    persist: bool = True,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> RiskOutput:
    """
    End-to-end rules pipeline for a case.

    Parameters
    ----------
    case_id       : unique case identifier
    case_entities : entity dict from extraction pipeline
    validations   : list of ValidationResult from validation_pipeline
    persist       : if True, write risk JSON to disk
    output_dir    : optional override for output directory
    config_path   : optional override for rules_config.yaml path

    Returns
    -------
    RiskOutput pydantic model
    """
    score, label, rule_hits = evaluate_rules(
        case_id=case_id,
        validations=validations,
        entities=case_entities,
        config_path=config_path,
    )

    if persist:
        persist_risk_score(case_id, score, label, rule_hits, output_dir)

    mandatory_review = any(h.mandatory_review for h in rule_hits)

    failed = [v for v in validations if not v.passed]
    summary_parts = [f"Score: {score} ({label.value})"]
    if mandatory_review:
        summary_parts.append("⚠️ Mandatory review required.")
    if failed:
        summary_parts.append(
            f"Issues: {', '.join(v.rule_id for v in failed)}"
        )

    return RiskOutput(
        case_id=case_id,
        risk_score=score,
        risk_label=label,
        mandatory_review=mandatory_review,
        rule_hits=rule_hits,
        validation_results=validations,
        summary=" | ".join(summary_parts),
    )