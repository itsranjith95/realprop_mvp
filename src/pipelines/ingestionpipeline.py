from sqlalchemy.orm import Session
from src.core.models import Document
from src.services.documentservice import save_uploaded_document


def run_ingestion(
    db: Session,
    case_id: str,
    doc_type: str,
    source_file_path: str,
    actor_id: str = "internal_user",
) -> Document:
    return save_uploaded_document(
        db=db,
        case_id=case_id,
        doc_type=doc_type,
        source_file_path=source_file_path,
        actor_id=actor_id,
    )