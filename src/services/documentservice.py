import mimetypes
from pathlib import Path

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.db_models import DocumentORM
from src.core.enums import AuditAction, CaseStatus, DocumentStatus, DocumentType
from src.core.models import Document
from src.core.utils import (
    copy_file,
    generate_id,
    sanitize_filename,
    sha256_for_file,
    utc_now,
)
from src.services.auditservice import log_audit_event
from src.services.caseservice import get_case, update_case_status

settings = get_settings()
ALLOWED_EXTENSIONS = set(settings["upload"]["allowed_extensions"])
MAX_FILE_SIZE_MB = int(settings["upload"]["max_file_size_mb"])
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def orm_to_document(row: DocumentORM) -> Document:
    return Document(
        id=row.id,
        case_id=row.case_id,
        doc_type=row.doc_type,
        original_filename=row.original_filename,
        stored_filename=row.stored_filename,
        file_extension=row.file_extension,
        mime_type=row.mime_type,
        file_size_bytes=row.file_size_bytes,
        version=row.version,
        checksum_sha256=row.checksum_sha256,
        path=row.path,
        status=row.status,
        created_at=row.created_at,
    )


def validate_file(filename: str, file_size_bytes: int) -> tuple[bool, str]:
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {suffix}"

    if file_size_bytes <= 0:
        return False, "Empty file is not allowed"

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        return False, f"File too large. Max allowed is {MAX_FILE_SIZE_MB} MB"

    return True, ""


def build_storage_path(case_id: str, doc_type: str, version: int, filename: str) -> Path:
    safe_name = sanitize_filename(filename)
    storage_dir = Path(settings["storage"]["raw_dir"]) / case_id / doc_type
    storage_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix.lower()
    stored_filename = f"{stem}_v{version}{suffix}"
    return storage_dir / stored_filename


def get_next_document_version(db: Session, case_id: str, doc_type: str) -> int:
    count = db.execute(
        select(func.count())
        .select_from(DocumentORM)
        .where(
            DocumentORM.case_id == case_id,
            DocumentORM.doc_type == doc_type,
        )
    ).scalar_one()
    return int(count) + 1


def save_uploaded_document(
    db: Session,
    case_id: str,
    doc_type: str,
    source_file_path: str,
    actor_id: str = "internal_user",
) -> Document:
    case = get_case(db, case_id)
    if not case:
        raise ValueError(f"Case not found: {case_id}")

    normalized_doc_type = DocumentType(doc_type).value
    source_path = Path(source_file_path)

    if not source_path.exists():
        raise ValueError(f"Source file not found: {source_file_path}")

    file_size_bytes = source_path.stat().st_size
    ok, err = validate_file(source_path.name, file_size_bytes)
    if not ok:
        log_audit_event(
            db=db,
            actor_id=actor_id,
            action_name=AuditAction.DOCUMENT_REJECTED.value,
            object_type="document",
            object_id=source_path.name,
            metadata={
                "case_id": case_id,
                "doc_type": normalized_doc_type,
                "reason": err,
            },
        )
        raise ValueError(err)

    version = get_next_document_version(db, case_id, normalized_doc_type)
    destination_path = build_storage_path(case_id, normalized_doc_type, version, source_path.name)
    copy_file(source_path, destination_path)

    checksum = sha256_for_file(destination_path)
    mime_type = mimetypes.guess_type(destination_path.name)[0] or "application/octet-stream"

    row = DocumentORM(
        id=generate_id("doc"),
        case_id=case_id,
        doc_type=normalized_doc_type,
        original_filename=source_path.name,
        stored_filename=destination_path.name,
        file_extension=destination_path.suffix.lower(),
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
        version=version,
        checksum_sha256=checksum,
        path=str(destination_path),
        status=DocumentStatus.UPLOADED.value,
        created_at=utc_now(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    log_audit_event(
        db=db,
        actor_id=actor_id,
        action_name=AuditAction.DOCUMENT_UPLOADED.value,
        object_type="document",
        object_id=row.id,
        metadata={
            "case_id": case_id,
            "doc_type": normalized_doc_type,
            "original_filename": row.original_filename,
            "stored_filename": row.stored_filename,
            "file_size_bytes": row.file_size_bytes,
            "version": row.version,
            "path": row.path,
        },
    )

    update_case_status(
        db=db,
        case_id=case_id,
        status=CaseStatus.UPLOADED,
        actor_id=actor_id,
    )

    return orm_to_document(row)


def list_documents_for_case(db: Session, case_id: str) -> list[Document]:
    rows = db.execute(
        select(DocumentORM)
        .where(DocumentORM.case_id == case_id)
        .order_by(DocumentORM.created_at.asc())
    ).scalars().all()
    return [orm_to_document(row) for row in rows]