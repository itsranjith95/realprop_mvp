from datetime import datetime
from pydantic import BaseModel, Field
from src.core.enums import CaseStatus, DocumentStatus, DocumentType
from typing import Any


class Case(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    status: CaseStatus = CaseStatus.DRAFT
    city: str = "Bengaluru"
    property_type: str = "residential"
    property_description: str | None = None
    created_by: str = "internal_user"


class Document(BaseModel):
    id: str
    case_id: str
    doc_type: DocumentType
    original_filename: str
    stored_filename: str
    file_extension: str
    mime_type: str
    file_size_bytes: int
    version: int = 1
    checksum_sha256: str
    path: str
    status: DocumentStatus = DocumentStatus.UPLOADED
    created_at: datetime


class AuditEvent(BaseModel):
    id: str
    actor_id: str
    action_name: str
    object_type: str
    object_id: str
    metadata_json: str
    created_at: datetime


class CreateCaseRequest(BaseModel):
    case_id: str | None = None
    property_description: str | None = None
    property_type: str = "residential"
    city: str = "Bengaluru"
    created_by: str = "internal_user"


class UpdateCaseStatusRequest(BaseModel):
    status: CaseStatus


class CaseResponse(BaseModel):
    case: Case


class DocumentResponse(BaseModel):
    document: Document


class CaseDetailResponse(BaseModel):
    case: Case
    documents: list[Document] = Field(default_factory=list)
    

class OCRBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float]
    block_type: str = "line"

class PageQuality(BaseModel):
    blur_score: float = 0.0
    contrast_score: float = 0.0
    brightness: float = 0.0
    skew_angle: float = 0.0
    quality_score: float = 0.0
    warnings: list[str] = []

class OCRPageResult(BaseModel):
    case_id: str
    document_id: str
    page_index: int
    source_file: str
    image_width: int
    image_height: int
    quality: PageQuality
    engine: str
    language: str
    page_confidence: float
    text: str
    blocks: list[OCRBlock]
    raw_result: dict[str, Any] = {}
    processed_at: str
    
# ─── Phase 4 additions ────────────────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    """Structured record for a single extracted & normalised field."""
    entity_id: str
    case_id: str
    document_id: str
    doc_type: str                        # "motherdeed" | "khata"
    field_name: str                      # e.g. "seller_name", "khata_number"
    value: str                           # raw extracted value
    normalized_value: str                # after normalisation
    confidence: float = Field(ge=0.0, le=1.0)
    page: int = 0
    bbox: list[float] = Field(default_factory=list)
    source_doc: str = ""
    extraction_method: str = "regex"     # "regex" | "spacy" | "llm"
    created_at: str = ""