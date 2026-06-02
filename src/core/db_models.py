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

    id               = Column(String(64),  primary_key=True)
    case_id          = Column(String(64),  ForeignKey("cases.id"), nullable=False, index=True)
    document_id      = Column(String(64),  ForeignKey("documents.id"), nullable=False, index=True)
    doc_type         = Column(String(50),  nullable=False)
    field_name       = Column(String(100), nullable=False)
    value            = Column(Text,        nullable=False)
    normalized_value = Column(Text,        nullable=False)
    confidence       = Column(String(10),  nullable=False)   # stored as str, cast to float in service
    page             = Column(Integer,     nullable=False, default=0)
    bbox             = Column(Text,        nullable=True)     # JSON list
    source_doc       = Column(String(255), nullable=True)
    extraction_method= Column(String(50),  nullable=False, default="regex")
    created_at       = Column(DateTime(timezone=True), nullable=False)