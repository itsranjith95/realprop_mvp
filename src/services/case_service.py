from src.core.models import Case
from src.core.utils import utc_now

def create_case(case_id: str, property_description: str | None = None) -> Case:
    return Case(
        id=case_id,
        created_at=utc_now(),
        property_description=property_description,
    )
