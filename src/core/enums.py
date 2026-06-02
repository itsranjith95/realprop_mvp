from enum import Enum


class DocumentType(str, Enum):
    MOTHER_DEED = "motherdeed"
    KHATA = "khata"

class DocumentStatus(str, Enum):
    # Phase 1/2 OCR pipeline statuses
    UPLOADED        = "UPLOADED"
    PROCESSING      = "PROCESSING"
    OCR_DONE        = "OCR_DONE"
    FAILED          = "FAILED"
    # Phase 2 validation statuses
    VALIDATED       = "validated"
    REJECTED        = "rejected"
    READY_FOR_OCR   = "ready_for_ocr"

class CaseStatus(str, Enum):
    DRAFT = "draft"
    AWAITING_DOCUMENTS = "awaiting_documents"
    INTAKE_VALIDATION = "intake_validation"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    REVIEW_REQUIRED = "review_required"
    REVIEWED = "reviewed"
    ERROR = "error"


class AuditAction(str, Enum):
    CASE_CREATED = "case_created"
    CASE_STATUS_UPDATED = "case_status_updated"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_REJECTED = "document_rejected"
    
    
# ─── Phase 4 additions ────────────────────────────────────────────────────────

class ExtractionMethod(str, Enum):
    REGEX = "regex"
    SPACY = "spacy"
    LLM   = "llm"

class MotherDeedField(str, Enum):
    SELLER_NAME          = "seller_name"
    BUYER_NAME           = "buyer_name"
    EXECUTION_DATE       = "execution_date"
    REGISTRATION_DATE    = "registration_date"
    PROPERTY_DESCRIPTION = "property_description"
    SURVEY_NUMBER        = "survey_number"
    SITE_NUMBER          = "site_number"
    FLAT_NUMBER          = "flat_number"
    REGISTRATION_OFFICE  = "registration_office"
    DOCUMENT_NUMBER      = "document_number"

class KhataField(str, Enum):
    OWNER_NAME   = "owner_name"
    KHATA_NUMBER = "khata_number"
    PROPERTY_ID  = "property_id"
    WARD         = "ward"
    ZONE         = "zone"
    ASSESSMENT   = "assessment"
    USAGE        = "usage"
    ADDRESS      = "address"