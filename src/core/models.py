from pydantic import BaseModel
from datetime import datetime
from .enums import CaseStatus, DocumentStatus, DocumentType

class Case(BaseModel):
    id: str
    created_at: datetime
    status: CaseStatus = CaseStatus.DRAFT
    city: str = "Bengaluru"
    property_description: str | None = None

class Document(BaseModel):
    id: str
    case_id: str
    doc_type: DocumentType
    path: str
    status: DocumentStatus = DocumentStatus.UPLOADED
