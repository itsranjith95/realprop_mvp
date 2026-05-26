from sqlalchemy.orm import Session
from src.core.models import Case
from src.services.caseservice import create_case


def run_intake(
    db: Session,
    case_id: str | None = None,
    property_description: str | None = None,
    property_type: str = "residential",
    city: str = "Bengaluru",
    created_by: str = "internal_user",
) -> Case:
    return create_case(
        db=db,
        case_id=case_id,
        property_description=property_description,
        property_type=property_type,
        city=city,
        created_by=created_by,
    )