from datetime import datetime
from pydantic import BaseModel, Field
from src.core.enums import CaseStatus, DocumentStatus, DocumentType


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