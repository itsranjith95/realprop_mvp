from enum import Enum

class DocumentType(str, Enum):
    MOTHER_DEED = "motherdeed"
    KHATA = "khata"

class CaseStatus(str, Enum):
    DRAFT = "draft"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    REVIEW_REQUIRED = "review_required"
    REVIEWED = "reviewed"

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    OCR_DONE = "ocr_done"
    EXTRACTED = "extracted"
