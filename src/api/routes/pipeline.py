import logging
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.classification import DocumentClassifier
from src.services.file_service import ensure_dir, is_allowed_file, new_id, save_upload
from src.services.ocr_pipeline import OCRPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])

RAW_DIR = Path("data/raw")
ocr_pipeline = OCRPipeline()
classifier = DocumentClassifier()

_pipeline_lock = threading.Lock()


def _aggregate_ocr_text(ocr_result: dict[str, Any]) -> str:
    pages = ocr_result.get("pages", [])
    chunks: list[str] = []

    for page in pages:
        if not isinstance(page, dict):
            continue

        if "text" in page and isinstance(page["text"], str):
            chunks.append(page["text"])
            continue

        if "blocks" in page and isinstance(page["blocks"], list):
            block_texts = []
            for block in page["blocks"]:
                if isinstance(block, dict):
                    text = block.get("text")
                    if text:
                        block_texts.append(str(text))
            if block_texts:
                chunks.append("\n".join(block_texts))
                continue

        if "raw_text" in page and isinstance(page["raw_text"], str):
            chunks.append(page["raw_text"])
            continue

        if "lines" in page and isinstance(page["lines"], list):
            line_texts = []
            for line in page["lines"]:
                if isinstance(line, dict):
                    text = line.get("text")
                    if text:
                        line_texts.append(str(text))
                elif isinstance(line, str):
                    line_texts.append(line)
            if line_texts:
                chunks.append("\n".join(line_texts))

    return "\n".join(c.strip() for c in chunks if c and c.strip()).strip()


@router.post("/ocr-classify", status_code=status.HTTP_200_OK)
def ocr_and_classify(
    case_id: str = Form(...),
    document_type: str = Form("unknown"),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    acquired = _pipeline_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="Another OCR/classification job is already running. Please wait and retry."
        )

    doc_id = new_id("doc")
    ext = Path(file.filename).suffix.lower()
    dst_dir = ensure_dir(RAW_DIR / case_id / document_type)
    dst_path = dst_dir / f"{doc_id}{ext}"

    try:
        save_upload(file, dst_path)

        start = time.perf_counter()
        logger.info("Starting OCR for doc_id=%s", doc_id)

        ocr_result = ocr_pipeline.process_document(case_id, doc_id, str(dst_path))
        logger.info("OCR finished for doc_id=%s in %.2fs", doc_id, time.perf_counter() - start)

        text_start = time.perf_counter()
        aggregated_text = _aggregate_ocr_text(ocr_result)
        logger.info("Text aggregation finished for doc_id=%s in %.2fs", doc_id, time.perf_counter() - text_start)

        clf_start = time.perf_counter()
        if not aggregated_text.strip():
            classification = classifier.classify("", doc_id=doc_id)
        else:
            classification = classifier.classify(aggregated_text, doc_id=doc_id)
        logger.info("Classification finished for doc_id=%s in %.2fs", doc_id, time.perf_counter() - clf_start)

        return {
            "case_id": case_id,
            "document_id": doc_id,
            "document_type_hint": document_type,
            "stored_path": str(dst_path),
            "ocr": ocr_result,
            "aggregated_text": aggregated_text,
            "classification": classification.to_dict(),
        }

    except Exception as exc:
        logger.exception("OCR + classification pipeline failed for doc_id=%s", doc_id)
        raise HTTPException(
            status_code=500,
            detail=f"pipeline failed: {str(exc)}"
        ) from exc
    finally:
        _pipeline_lock.release()