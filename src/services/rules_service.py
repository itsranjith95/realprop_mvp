from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

try:
    from rapidfuzz import fuzz
except ImportError:  # fallback if rapidfuzz is not installed
    fuzz = None

RULES_CONFIG_PATH = Path("config/rules_config.yaml")


def load_rules_config() -> Dict[str, Any]:
    """Load rules YAML config."""
    if not RULES_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Rules config not found: {RULES_CONFIG_PATH}")
    with open(RULES_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_value(field_obj: Any) -> str:
    if isinstance(field_obj, dict):
        return _norm(field_obj.get("value", ""))
    return _norm(field_obj)


def _get_conf(field_obj: Any) -> float:
    if isinstance(field_obj, dict):
        try:
            return float(field_obj.get("confidence", 0.0))
        except Exception:
            return 0.0
    return 0.0


def _fuzzy_score(a: str, b: str) -> int:
    a = _norm(a).lower()
    b = _norm(b).lower()
    if not a or not b:
        return 0
    if fuzz:
        return int(fuzz.ratio(a, b))
    return 100 if a == b else 0


def _risk_label(score: int, risk_cfg: Dict[str, Any]) -> str:
    green_max = int(risk_cfg.get("green_max", 15))
    yellow_max = int(risk_cfg.get("yellow_max", 40))
    labels = risk_cfg.get("labels", {})
    if score <= green_max:
        return labels.get("green", "Low Risk")
    if score <= yellow_max:
        return labels.get("yellow", "Medium Risk — Review Recommended")
    return labels.get("red", "High Risk — Mandatory Lawyer Review")


def evaluate_rules(case_id: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected entities shape:
    {
      "mother_deed": {
        "buyer_name": {"value": "...", "confidence": 0.95},
        ...
      },
      "khata": {
        "owner_name": {"value": "...", "confidence": 0.95},
        ...
      }
    }
    """
    config = load_rules_config()
    rules = config.get("rules", [])
    risk_cfg = config.get("risk_scoring", {})

    md = entities.get("mother_deed", {}) or {}
    kh = entities.get("khata", {}) or {}

    rule_hits: List[Dict[str, Any]] = []
    total_score = 0

    for rule in rules:
        rule_id = rule["rule_id"]
        triggered = False
        evidence = {}
        condition = rule.get("condition", "")

        # Single-field checks
        if condition == "missing_or_empty":
            field = rule.get("field", "")
            applies_to = rule.get("applies_to", [])

            if "MOTHER_DEED" in applies_to:
                val = _get_value(md.get(field))
                if not val:
                    triggered = True
                    evidence = {"document": "mother_deed", "field": field, "value": val}

            elif "KHATA" in applies_to:
                val = _get_value(kh.get(field))
                if not val:
                    triggered = True
                    evidence = {"document": "khata", "field": field, "value": val}

            elif set(applies_to) == {"MOTHER_DEED", "KHATA"} or ("MOTHER_DEED" in applies_to and "KHATA" in applies_to):
                md_val = _get_value(md.get(field))
                kh_val = _get_value(kh.get(field))
                if not md_val or not kh_val:
                    triggered = True
                    evidence = {"document": "cross", "field": field, "mother_deed": md_val, "khata": kh_val}

        # Cross exact mismatch
        elif condition == "exact_mismatch":
            fields = rule.get("fields", [])
            if len(fields) == 2:
                left = fields[0]
                right = fields[1]

                left_val = _get_value(md.get(left))
                if right == "khata_property_id":
                    right_val = _get_value(kh.get("property_id"))
                else:
                    right_val = _get_value(kh.get(right))

                if left_val and right_val and left_val != right_val:
                    triggered = True
                    evidence = {"left_field": left, "left_value": left_val, "right_field": right, "right_value": right_val}

        # Cross fuzzy mismatch
        elif condition == "fuzzy_mismatch":
            fields = rule.get("fields", [])
            threshold = int(rule.get("fuzzy_threshold", 85))
            if len(fields) == 2:
                left = fields[0]
                right = fields[1]

                left_val = _get_value(md.get(left))
                right_val = _get_value(kh.get(right))
                score = _fuzzy_score(left_val, right_val)

                if left_val and right_val and score < threshold:
                    triggered = True
                    evidence = {
                        "left_field": left,
                        "left_value": left_val,
                        "right_field": right,
                        "right_value": right_val,
                        "fuzzy_score": score,
                        "threshold": threshold,
                    }

        # Numeric mismatch with tolerance
        elif condition == "numeric_mismatch":
            fields = rule.get("fields", [])
            tolerance = float(rule.get("tolerance_percent", 5))
            if len(fields) == 2:
                left_val = _get_value(md.get(fields[0]))
                right_field = fields[1]
                if right_field == "khata_area":
                    right_val = _get_value(kh.get("property_area"))
                else:
                    right_val = _get_value(kh.get(right_field))
                try:
                    a = float(left_val)
                    b = float(right_val)
                    delta_pct = abs(a - b) / a * 100 if a else 0
                    if a and b and delta_pct > tolerance:
                        triggered = True
                        evidence = {"left_value": a, "right_value": b, "delta_pct": round(delta_pct, 2)}
                except Exception:
                    pass

        # Confidence checks
        elif condition == "confidence_below_threshold":
            threshold = float(rule.get("confidence_threshold", 0.75))
            critical_fields = rule.get("critical_fields", [])
            low_fields = []

            for f in critical_fields:
                if f in md and _get_conf(md.get(f)) < threshold:
                    low_fields.append(("mother_deed", f, _get_conf(md.get(f))))
                if f in kh and _get_conf(kh.get(f)) < threshold:
                    low_fields.append(("khata", f, _get_conf(kh.get(f))))

            if low_fields:
                triggered = True
                evidence = {"low_fields": low_fields, "threshold": threshold}

        elif condition == "page_ocr_confidence_below_threshold":
            # MVP placeholder: skip unless page-level OCR confidence is later passed in entities
            triggered = False

        if triggered:
            pts = int(rule.get("points", 0))
            total_score += pts
            rule_hits.append({
                "case_id": case_id,
                "rule_id": rule_id,
                "rule_version": rule.get("rule_version"),
                "rule_name": rule.get("name"),
                "severity": rule.get("severity"),
                "description": rule.get("description"),
                "points": pts,
                "requires_review": rule.get("requires_review", False),
                "evidence": evidence,
            })

    return {
        "case_id": case_id,
        "risk_score": total_score,
        "risk_label": _risk_label(total_score, risk_cfg),
        "rule_hits": rule_hits,
        "rules_evaluated": len(rules),
        "rule_set_version": config.get("meta", {}).get("rule_set_version"),
    }