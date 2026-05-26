from sqlalchemy.orm import Session
from src.core.db_models import AuditEventORM
from src.core.models import AuditEvent
from src.core.utils import generate_id, json_dumps, utc_now


def log_audit_event(
    db: Session,
    actor_id: str,
    action_name: str,
    object_type: str,
    object_id: str,
    metadata: dict,
) -> AuditEvent:
    row = AuditEventORM(
        id=generate_id("audit"),
        actor_id=actor_id,
        action_name=action_name,
        object_type=object_type,
        object_id=object_id,
        metadata_json=json_dumps(metadata),
        created_at=utc_now(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return AuditEvent(
        id=row.id,
        actor_id=row.actor_id,
        action_name=row.action_name,
        object_type=row.object_type,
        object_id=row.object_id,
        metadata_json=row.metadata_json,
        created_at=row.created_at,
    )