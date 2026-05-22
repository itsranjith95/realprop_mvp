from src.services.case_service import create_case

def run_intake(case_id: str, property_description: str | None = None):
    return create_case(case_id=case_id, property_description=property_description)
