from src.services.document_service import validate_file, build_storage_path, create_document_record

def run_ingestion(case_id: str, doc_type: str, filename: str):
    ok, err = validate_file(filename)
    if not ok:
        raise ValueError(err)

    path = build_storage_path(case_id, doc_type, filename)
    return create_document_record(case_id=case_id, doc_type=doc_type, path=str(path))
