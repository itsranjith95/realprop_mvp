from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from src.core.database import Base


class CaseORM(Base):
    __tablename__ = "cases"

    id = Column(String(64), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(50), nullable=False)
    city = Column(String(100), nullable=False)
    property_type = Column(String(100), nullable=False)
    property_description = Column(Text, nullable=True)
    created_by = Column(String(100), nullable=False)


class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True)
    case_id = Column(String(64), ForeignKey("cases.id"), nullable=False, index=True)
    doc_type = Column(String(50), nullable=False)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    file_extension = Column(String(20), nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    checksum_sha256 = Column(String(128), nullable=False)
    path = Column(Text, nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class AuditEventORM(Base):
    __tablename__ = "audit_events"

    id = Column(String(64), primary_key=True)
    actor_id = Column(String(100), nullable=False)
    action_name = Column(String(100), nullable=False)
    object_type = Column(String(100), nullable=False)
    object_id = Column(String(100), nullable=False, index=True)
    metadata_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


# ─── Phase 4 addition ─────────────────────────────────────────────────────────


class ExtractedEntityORM(Base):
    __tablename__ = "extracted_entities"

    id = Column(String(64), primary_key=True)
    case_id = Column(String(64), ForeignKey("cases.id"), nullable=False, index=True)
    document_id = Column(String(64), ForeignKey("documents.id"), nullable=False, index=True)
    doc_type = Column(String(50), nullable=False)
    field_name = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)
    normalized_value = Column(Text, nullable=False)
    confidence = Column(String(10), nullable=False)  # stored as str, cast to float in service
    page = Column(Integer, nullable=False, default=0)
    bbox = Column(Text, nullable=True)  # JSON list
    source_doc = Column(String(255), nullable=True)
    extraction_method = Column(String(50), nullable=False, default="regex")
    created_at = Column(DateTime(timezone=True), nullable=False)


# ── Phase 7 addition ──────────────────────────────────────────────────────────


class LawyerReview(Base):
    """
    Persists lawyer review actions in SQLite.
    Note: review_pipeline.py also writes JSON files to data/reviews/
    This table is the canonical source for API queries.
    """
    __tablename__ = "lawyer_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    action = Column(String, nullable=False)  # approve | request_clarification | mark_high_risk
    notes = Column(Text, default="")
    lawyer_name = Column(String, default="Lawyer")
    final_label = Column(String, default="")
    reviewed_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "action": self.action,
            "notes": self.notes,
            "lawyer_name": self.lawyer_name,
            "final_label": self.final_label,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else "",
        }