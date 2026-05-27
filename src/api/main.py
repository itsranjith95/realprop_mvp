from pathlib import Path
import shutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from src.services.file_service import ensure_dir, is_allowed_file, new_id
from src.services.ocr_pipeline import OCRPipeline

app = FastAPI(title="RealProp Phase 2 OCR API")
pipeline = OCRPipeline()
RAW_DIR = Path("data/raw")

@app.get("/health")
def health():
    return {"status": "ok", "phase": 2}

@app.post("/v1/ocr/run")
def run_ocr(case_id: str = Form(...), document_type: str = Form("unknown"), file: UploadFile = File(...)):
    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    doc_id = new_id("doc")
    dst_dir = ensure_dir(RAW_DIR / case_id / document_type)
    dst = dst_dir / f"{doc_id}{Path(file.filename).suffix.lower()}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    result = pipeline.process_document(case_id, doc_id, str(dst))
    return {"case_id": case_id, "document_id": doc_id, "stored_path": str(dst), "result": result}