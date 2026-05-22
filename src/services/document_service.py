from pathlib import Path
from src.core.models import Document
from src.core.enums import DocumentType
from src.core.utils import generate_id, ensure_dir

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

def validate_file(filename: str):
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {suffix}"
    return True, ""

def build_storage_path(case_id: str, doc_type: str, filename: str) -> Path:
    folder = ensure_dir(Path("data/raw") / case_id / doc_type)
    return folder / filename

def create_document_record(case_id: str, doc_type: str, path: str) -> Document:
    return Document(
        id=generate_id("doc"),
        case_id=case_id,
        doc_type=DocumentType(doc_type),
        path=path,
    )
