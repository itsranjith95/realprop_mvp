from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.db_models import CaseORM
from src.core.enums import AuditAction, CaseStatus
from src.core.models import Case
from src.core.utils import generate_id, utc_now
from src.services.auditservice import log_audit_event


def orm_to_case(row: CaseORM) -> Case:
    return Case(
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        status=row.status,
        city=row.city,
        property_type=row.property_type,
        property_description=row.property_description,
        created_by=row.created_by,
    )


def create_case(
    db: Session,
    case_id: str | None = None,
    property_description: str | None = None,
    property_type: str = "residential",
    city: str = "Bengaluru",
    created_by: str = "internal_user",
) -> Case:
    final_case_id = case_id or generate_id("case")
    existing = db.get(CaseORM, final_case_id)
    if existing:
        raise ValueError(f"Case already exists: {final_case_id}")

    now = utc_now()
    row = CaseORM(
        id=final_case_id,
        created_at=now,
        updated_at=now,
        status=CaseStatus.DRAFT.value,
        city=city,
        property_type=property_type,
        property_description=property_description,
        created_by=created_by,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    log_audit_event(
        db=db,
        actor_id=created_by,
        action_name=AuditAction.CASE_CREATED.value,
        object_type="case",
        object_id=final_case_id,
        metadata={
            "city": city,
            "property_type": property_type,
            "status": CaseStatus.DRAFT.value,
        },
    )

    return orm_to_case(row)


def get_case(db: Session, case_id: str) -> Case | None:
    row = db.get(CaseORM, case_id)
    if not row:
        return None
    return orm_to_case(row)


def list_cases(db: Session) -> list[Case]:
    rows = db.execute(select(CaseORM).order_by(CaseORM.created_at.desc())).scalars().all()
    return [orm_to_case(row) for row in rows]


def update_case_status(
    db: Session,
    case_id: str,
    status: CaseStatus,
    actor_id: str = "system",
) -> Case:
    row = db.get(CaseORM, case_id)
    if not row:
        raise ValueError(f"Case not found: {case_id}")

    old_status = row.status
    row.status = status.value
    row.updated_at = utc_now()
    db.commit()
    db.refresh(row)

    log_audit_event(
        db=db,
        actor_id=actor_id,
        action_name=AuditAction.CASE_STATUS_UPDATED.value,
        object_type="case",
        object_id=case_id,
        metadata={
            "old_status": old_status,
            "new_status": status.value,
        },
    )

    return orm_to_case(row)

def update_case_status(case_id: str, status: str) -> bool:
    """
    Update the status of a case in the SQLite DB.
    Called by Phase 7 when a lawyer approves / marks high risk.

    Parameters
    ----------
    case_id : str
    status  : e.g. 'REVIEW_COMPLETE', 'HIGH_RISK', 'NEEDS_CLARIFICATION'

    Returns
    -------
    bool – True if a row was updated, False if case_id not found
    """
    import sqlite3
    from src.pipelines.review_pipeline import DB_PATH

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "UPDATE cases SET status = ? WHERE id = ?",
            (status, case_id),
        )
        affected = cur.rowcount
        conn.commit()
        conn.close()
        return affected > 0
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(
            "Failed to update case status for %s: %s", case_id, exc
        )
        return False