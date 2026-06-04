"""
src/app/api.py
FastAPI backend for RealProp MVP.
Endpoints:
  GET  /health
  POST /api/v1/pipeline/ocr-classify         (existing – Phase 3)
  POST /api/v1/pipeline/full-process          (NEW – Phase 3→5→6, writes risk JSON)
  POST /api/v1/runtime/classify-manifest
  POST /api/v1/runtime/save-review-example
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("realprop.api")

app = FastAPI(title="RealProp MVP API", version="1.0.0")

# ── pipeline lock (single-user MVP) ──────────────────────────────────────────
_pipeline_running = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(upload.file, fh)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "RealProp MVP API"}


@app.get("/api/v1/classify/health")
def classify_health():
    try:
        from src.services.classificationservice import ClassificationService
        svc = ClassificationService()
        return {"status": "ok", "classifier": str(svc)}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# OCR + Classify  (Phase 3 – existing)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/pipeline/ocr-classify")
async def ocr_classify(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    document_type: str = Form("unknown"),
):
    global _pipeline_running
    if _pipeline_running:
        raise HTTPException(status_code=409, detail="Another pipeline job is already running.")
    _pipeline_running = True

    try:
        from src.pipelines.extraction_pipeline import run_extraction_pipeline
        from src.services.ocr_service import OCRService
        from src.services.classificationservice import ClassificationService

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        ocr_svc = OCRService()
        ocr_result = ocr_svc.run_ocr(tmp_path, case_id=case_id)

        cls_svc = ClassificationService()
        aggregated_text = " ".join(
            p.get("text", "") for p in ocr_result.get("pages", [])
        )
        classification = cls_svc.predict(aggregated_text)

        tmp_path.unlink(missing_ok=True)

        return JSONResponse({
            "case_id": case_id,
            "document_id": ocr_result.get("document_id", ""),
            "aggregated_text": aggregated_text,
            "ocr": ocr_result,
            "classification": classification,
        })

    except Exception as exc:
        logger.exception("ocr-classify failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _pipeline_running = False


# ─────────────────────────────────────────────────────────────────────────────
# FULL PROCESS  (NEW – Phase 3 → 4 → 5 → 6, writes risk JSON)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/pipeline/full-process")
async def full_process(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    document_type: str = Form("unknown"),
):
    """
    Run the complete processing pipeline for one document:
      1. OCR
      2. Classification
      3. Entity Extraction  (writes to SQLite extracted_entities table)
      4. Validation         (cross-doc checks)
      5. Rules + Risk Score (writes data/results/risk_scores/{case_id}_risk.json)

    Returns the RiskOutput as JSON so the UI can display it immediately.
    """
    global _pipeline_running
    if _pipeline_running:
        raise HTTPException(status_code=409, detail="Another pipeline job is already running.")
    _pipeline_running = True

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        # ── Step 1: OCR ──────────────────────────────────────────────────────
        from src.services.ocrservice import OCRService
        ocr_svc    = OCRService()
        ocr_result = ocr_svc.run_ocr(tmp_path, case_id=case_id)
        aggregated_text = " ".join(
            p.get("text", "") for p in ocr_result.get("pages", [])
        )

        # ── Step 2: Classification ────────────────────────────────────────────
        from src.services.classificationservice import ClassificationService
        cls_svc        = ClassificationService()
        classification = cls_svc.predict(aggregated_text)
        doc_type       = classification.get("doc_type", document_type)

        # ── Step 3: Extraction ────────────────────────────────────────────────
        from src.pipelines.extraction_pipeline import run_extraction_pipeline
        extraction_result = run_extraction_pipeline(
            case_id=case_id,
            doc_id=ocr_result.get("document_id", ""),
            doc_type=doc_type,
            ocr_result=ocr_result,
        )

        # ── Step 4: Validation ────────────────────────────────────────────────
        from src.pipelines.validation_pipeline import collect_validation_results
        case_entities = extraction_result.get("entities", {})
        validations   = collect_validation_results(case_id, case_entities)

        # ── Step 5: Rules + Risk Score ────────────────────────────────────────
        from src.pipelines.rules_pipeline import run_rules_pipeline
        risk_output = run_rules_pipeline(
            case_id=case_id,
            case_entities=case_entities,
            validations=validations,
            persist=True,          # ← this writes data/results/risk_scores/{case_id}_risk.json
        )

        tmp_path.unlink(missing_ok=True)

        return JSONResponse({
            "case_id":        case_id,
            "document_id":    ocr_result.get("document_id", ""),
            "doc_type":       doc_type,
            "aggregated_text": aggregated_text,
            "classification": classification,
            "risk_score":     risk_output.risk_score,
            "risk_label":     risk_output.risk_label.value,
            "risk_summary":   risk_output.summary,
            "mandatory_review": risk_output.mandatory_review,
            "rule_hits":      [h.model_dump() for h in risk_output.rule_hits],
        })

    except Exception as exc:
        logger.exception("full-process failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _pipeline_running = False


# ─────────────────────────────────────────────────────────────────────────────
# Classify from manifest  (existing)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/runtime/classify-manifest")
async def classify_manifest(body: dict):
    manifest_path = body.get("manifest_path", "")
    mp = Path(manifest_path)
    if not mp.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest_path}")
    try:
        from src.services.classificationservice import ClassificationService
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        pages    = manifest.get("pages", [])
        aggregated_text = " ".join(p.get("text", "") for p in pages)
        cls_svc        = ClassificationService()
        classification = cls_svc.predict(aggregated_text)
        return JSONResponse({
            "manifest_path": manifest_path,
            "aggregated_text": aggregated_text,
            "classification": classification,
        })
    except Exception as exc:
        logger.exception("classify-manifest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Save review example  (existing)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/runtime/save-review-example")
async def save_review_example(body: dict):
    try:
        from src.services.classificationservice import ClassificationService
        svc = ClassificationService()
        result = svc.save_review_example(
            case_id=body.get("case_id", ""),
            document_id=body.get("document_id", ""),
            confirmed_label=body.get("confirmed_label", ""),
            aggregated_text=body.get("aggregated_text", ""),
            source=body.get("source", "manual_review"),
            review_notes=body.get("review_notes", ""),
        )
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("save-review-example failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))