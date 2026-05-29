"""
POST /api/v1/classify
POST /api/v1/classify/batch
GET  /api/v1/classify/health
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.classification import DocumentClassifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/classify", tags=["classification"])

# Singleton classifier — initialised once at module load
_classifier: Optional[DocumentClassifier] = None


def get_classifier() -> DocumentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier()
    return _classifier


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="OCR extracted text from Phase 2")
    doc_id: Optional[str] = Field(None, description="Optional document identifier")


class ClassifyResponse(BaseModel):
    doc_id: Optional[str]
    doc_type: str
    confidence: float
    method: str
    matched_keywords: list[str]
    reasoning: str
    rule_score: float
    ollama_score: float
    needs_human_review: bool
    all_scores: dict
    error: Optional[str]


class BatchClassifyRequest(BaseModel):
    documents: list[ClassifyRequest]


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    total: int
    needs_review_count: int


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_model: str
    available_models: list[str]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def classification_health():
    """Check classifier and Ollama availability."""
    clf = get_classifier()
    ollama_available = clf.ollama.is_available() if clf.ollama else False
    models = clf.ollama.get_available_models() if clf.ollama else []
    return HealthResponse(
        status="ok",
        ollama_available=ollama_available,
        ollama_model=clf.ollama.model if clf.ollama else "none",
        available_models=models,
    )


@router.post("/", response_model=ClassifyResponse, status_code=status.HTTP_200_OK)
def classify_document(request: ClassifyRequest):
    """Classify a single document from its extracted text."""
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="text field cannot be empty",
        )
    clf = get_classifier()
    result = clf.classify(request.text, doc_id=request.doc_id)
    return ClassifyResponse(doc_id=request.doc_id, **result.to_dict())


@router.post("/batch", response_model=BatchClassifyResponse)
def classify_batch(request: BatchClassifyRequest):
    """Classify multiple documents in one call."""
    if not request.documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="documents list cannot be empty",
        )
    clf = get_classifier()
    texts = [{"doc_id": d.doc_id, "text": d.text} for d in request.documents]
    results = clf.classify_batch(texts)

    responses = [ClassifyResponse(**r) for r in results]
    needs_review = sum(1 for r in responses if r.needs_human_review)

    return BatchClassifyResponse(
        results=responses,
        total=len(responses),
        needs_review_count=needs_review,
    )