from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile

from src.core.database import SessionLocal
from src.core.init_db import init_db
from src.core.models import CreateCaseRequest, UpdateCaseStatusRequest, CaseDetailResponse, CaseResponse, DocumentResponse
from src.core.enums import CaseStatus
from src.pipelines.intakepipeline import run_intake
from src.pipelines.ingestionpipeline import run_ingestion
from src.services.caseservice import get_case, list_cases, update_case_status
from src.services.documentservice import list_documents_for_case

app = FastAPI(title="RealProp MVP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/cases", response_model=CaseResponse)
def create_case(payload: CreateCaseRequest):
    db = SessionLocal()
    try:
        case = run_intake(
            db=db,
            case_id=payload.case_id,
            property_description=payload.property_description,
            property_type=payload.property_type,
            city=payload.city,
            created_by=payload.created_by,
        )
        return {"case": case}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@app.get("/api/v1/cases")
def get_cases():
    db = SessionLocal()
    try:
        cases = list_cases(db)
        return {"cases": cases}
    finally:
        db.close()


@app.get("/api/v1/cases/{case_id}", response_model=CaseDetailResponse)
def get_case_detail(case_id: str):
    db = SessionLocal()
    try:
        case = get_case(db, case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        documents = list_documents_for_case(db, case_id)
        return {"case": case, "documents": documents}
    finally:
        db.close()


@app.patch("/api/v1/cases/{case_id}/status", response_model=CaseResponse)
def patch_case_status(case_id: str, payload: UpdateCaseStatusRequest):
    db = SessionLocal()
    try:
        case = update_case_status(db, case_id, payload.status, actor_id="streamlit_user")
        return {"case": case}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        db.close()


@app.post("/api/v1/cases/{case_id}/documents", response_model=DocumentResponse)
async def upload_document(
    case_id: str,
    doc_type: str = Form(...),
    actor_id: str = Form("streamlit_user"),
    file: UploadFile = File(...),
):
    db = SessionLocal()
    temp_path = None
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        document = run_ingestion(
            db=db,
            case_id=case_id,
            doc_type=doc_type,
            source_file_path=temp_path,
            actor_id=actor_id,
        )
        return {"document": document}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)