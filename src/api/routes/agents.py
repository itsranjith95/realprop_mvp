"""
src/api/routes/agents.py
Phase 6 – Agentic AI Layer – FastAPI route

Exposes POST /v1/agents/{case_id}/run
Accepts extracted entities + optional OCR confidence and risk context.
Returns AgentOutputs as JSON.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.validation_models import RiskOutput, ValidationResult
from src.services.agent_service import AgentOutputs, run_agents

router = APIRouter(prefix="/v1/agents", tags=["agents"])


class RunAgentsRequest(BaseModel):
    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted entity dictionaries from OCR/extraction pipeline",
    )
    ocr_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall OCR confidence score from 0 to 1",
    )
    rule_outputs: list[ValidationResult] | None = Field(
        default=None,
        description="Optional Phase 5 validation results",
    )
    risk_output: RiskOutput | None = Field(
        default=None,
        description="Optional Phase 5 risk output",
    )


class RunAgentsResponse(BaseModel):
    case_id: str
    agent_outputs: AgentOutputs


@router.post(
    "/{case_id}/run",
    response_model=RunAgentsResponse,
    summary="Run all Phase 6 agents for a case",
    description=(
        "Runs ExtractionValidator, TitleChain, KhataAnalysis, Compliance, "
        "RiskSynthesis, and LLM explanation generation for a case."
    ),
)
def run_agents_endpoint(case_id: str, body: RunAgentsRequest) -> RunAgentsResponse:
    if not case_id or not case_id.strip():
        raise HTTPException(status_code=400, detail="case_id must not be empty")

    outputs = run_agents(
        case_id=case_id.strip(),
        entities=body.entities,
        ocr_confidence=body.ocr_confidence,
        rule_outputs=body.rule_outputs,
        risk_output=body.risk_output,
    )

    return RunAgentsResponse(
        case_id=case_id.strip(),
        agent_outputs=outputs,
    )