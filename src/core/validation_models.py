# src/core/validation_models.py
"""
Pydantic data-models shared across Phase 5 pipelines.
ValidationResult  → output of each individual check
RiskOutput        → final scored output for a case
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RiskLabel(str, Enum):
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"


class ValidationType(str, Enum):
    OWNER_NAME = "OWNER_NAME"
    PROPERTY_ID = "PROPERTY_ID"
    AREA = "AREA"
    MISSING_FIELD = "MISSING_FIELD"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Single cross-document validation finding."""

    rule_id: str = Field(..., description="Unique rule identifier, e.g. OWNER_NAME_MISMATCH")
    type: ValidationType = Field(..., description="Broad category of the validation check")
    severity: Severity = Field(..., description="Severity level of the finding")
    description: str = Field(..., description="Human-readable explanation of the finding")
    passed: bool = Field(
        ...,
        description="True = check passed (no issue), False = check failed (issue found)",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value evidence linking to source fields",
    )
    mandatory_review: bool = Field(
        default=False,
        description="If True, escalate to manual review regardless of total risk score",
    )


class RuleHit(BaseModel):
    """A triggered governance rule with its evidence."""

    rule_id: str
    name: str
    severity: Severity
    points: int
    mandatory_review: bool
    evidence: Dict[str, Any] = Field(default_factory=dict)


class RiskOutput(BaseModel):
    """Final risk-scored output for a single case."""

    case_id: str
    risk_score: int = Field(..., ge=0, le=100, description="Deterministic risk score 0–100")
    risk_label: RiskLabel
    mandatory_review: bool = Field(
        default=False,
        description="True if any triggered rule demands manual review",
    )
    rule_hits: List[RuleHit] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    summary: Optional[str] = None