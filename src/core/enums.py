from enum import Enum


class DocumentType(str, Enum):
    MOTHER_DEED = "motherdeed"
    KHATA = "khata"


class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    OCR_DONE = "OCR_DONE"
    FAILED = "FAILED"

class CaseStatus(str, Enum):
    DRAFT = "draft"
    AWAITING_DOCUMENTS = "awaiting_documents"
    INTAKE_VALIDATION = "intake_validation"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    REVIEW_REQUIRED = "review_required"
    REVIEWED = "reviewed"
    ERROR = "error"


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    VALIDATED = "validated"
    REJECTED = "rejected"
    READY_FOR_OCR = "ready_for_ocr"


class AuditAction(str, Enum):
    CASE_CREATED = "case_created"
    CASE_STATUS_UPDATED = "case_status_updated"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_REJECTED = "document_rejected"