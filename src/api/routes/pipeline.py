import json
import logging
import re
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

    return "\n".join(c.strip() for c in chunks if c and c.strip()).strip()


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_import_full_process_helpers():
    try:
        from src.pipelines.extraction_pipeline import run_extraction_pipeline
    except Exception:
        run_extraction_pipeline = None

    try:
        from src.pipelines.validation_pipeline import run_validation_pipeline
    except Exception:
        run_validation_pipeline = None

    try:
        from src.pipelines.rules_pipeline import run_rules_pipeline
    except Exception:
        run_rules_pipeline = None

    return run_extraction_pipeline, run_validation_pipeline, run_rules_pipeline


def _extract_fallback_entities(text: str, case_id: str, doc_id: str, doc_type: str) -> dict:
    seller = None
    buyer = None
    deed_date = None
    property_no = None
    area_text = None

    clean_text = re.sub(r"\s+", " ", text)

    seller_patterns = [
        r"By:\s*Sri\.?\s*([A-Za-z\.\s]+?),\s*son of",
        r"By:\s*Smt\.?\s*([A-Za-z\.\s]+?),\s*wife of",
        r"Sri\.?\s*([A-Za-z\.\s]+?),\s*son of",
        r"Smt\.?\s*([A-Za-z\.\s]+?),\s*wife of",
        r"Smt\.?\s*([A-Za-z\.\s]+?),\s*W/o",
        r"Sri\.?\s*([A-Za-z\.\s]+?),\s*S/o",
    ]
    for pat in seller_patterns:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            seller = m.group(1).strip()
            break

    buyer_patterns = [
        r"In favour of:\s*Smt\.?\s*([A-Za-z\.\s]+?),\s*wife of",
        r"In favour of:\s*Sri\.?\s*([A-Za-z\.\s]+?),\s*son of",
        r"PURCHASER\.\s*The expressions",
        r"Smt\.?\s*([A-Za-z\.\s]+?),\s*wife of",
        r"Sri\.?\s*([A-Za-z\.\s]+?),\s*son of",
    ]
    for i, pat in enumerate(buyer_patterns):
        m = re.search(pat, clean_text, re.IGNORECASE)
        if not m:
            continue
        if i == 2:
            pass
        else:
            candidate = m.group(1).strip()
            if seller and candidate.lower() == seller.lower():
                continue
            buyer = candidate
            break

    date_match = re.search(
        r"(?:made and executed on|made and executed at|executed on)\s+this\s+(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+\s+\d{4})",
        clean_text,
        re.IGNORECASE,
    )
    if date_match:
        deed_date = date_match.group(1).strip()

    property_match = re.search(r"Property\s+No\.?\s*([A-Za-z0-9\-\/]+)", clean_text, re.IGNORECASE)
    if property_match:
        property_no = property_match.group(1).strip()

    area_match = re.search(r"measuring\s+([^.]+)", clean_text, re.IGNORECASE)
    if area_match:
        area_text = area_match.group(1).strip()

    fields = {
        "document_type": {"value": doc_type, "confidence": 0.95},
        "seller_name": {"value": seller or "", "confidence": 0.82 if seller else 0.0},
        "buyer_name": {"value": buyer or "", "confidence": 0.82 if buyer else 0.0},
        "execution_date": {"value": deed_date or "", "confidence": 0.75 if deed_date else 0.0},
        "property_number": {"value": property_no or "", "confidence": 0.68 if property_no else 0.0},
        "property_area": {"value": area_text or "", "confidence": 0.50 if area_text else 0.0},
    }

    return {
        "case_id": case_id,
        "document_id": doc_id,
        "doc_type": doc_type,
        "fields": fields,
        "extraction_method": "fallback_regex",
    }


def _override_doc_type_from_text(predicted_label: str, aggregated_text: str) -> str:
    t = aggregated_text.lower()

    strong_mother_deed_terms = [
        "mother deed",
        "parent title deed",
        "parent deed",
        "origin deed",
        "chain-of-title",
        "chain of title",
    ]

    strong_sale_deed_terms = [
        "sale deed",
        "deed of absolute sale",
    ]

    md_hits = sum(1 for term in strong_mother_deed_terms if term in t)
    sd_hits = sum(1 for term in strong_sale_deed_terms if term in t)

    if md_hits >= 2:
        return "mother_deed"

    if sd_hits >= 2 and md_hits == 0:
        return "sale_deed"

    return predicted_label


def _build_fallback_validation(extraction_result: dict) -> dict:
    fields = extraction_result.get("fields", {})
    flags = []

    critical_fields = ["seller_name", "buyer_name", "execution_date", "property_number"]
    for field_name in critical_fields:
        field = fields.get(field_name, {})
        value = str(field.get("value", "")).strip()
        conf = float(field.get("confidence", 0.0) or 0.0)

        if not value:
            flags.append({
                "rule": f"MISSING_{field_name.upper()}",
                "status": "FAIL",
                "detail": f"{field_name} not found in extracted fields",
                "severity": "high",
            })
        elif conf < 0.65:
            flags.append({
                "rule": f"LOW_CONFIDENCE_{field_name.upper()}",
                "status": "WARN",
                "detail": f"{field_name} extracted with low confidence ({conf:.2f})",
                "severity": "medium",
            })

    overall_status = "PASS" if not any(f["status"] == "FAIL" for f in flags) else "FAIL"

    return {
        "overall_status": overall_status,
        "flags": flags,
        "validation_method": "fallback_rules",
    }


def _build_fallback_rules(case_id: str, validation_result: dict, classification_dict: dict) -> dict:
    flags = validation_result.get("flags", [])
    score = 10

    for flag in flags:
        sev = str(flag.get("severity", "")).lower()
        status = str(flag.get("status", "")).lower()
        if sev == "high" and status == "fail":
            score += 30
        elif sev == "medium":
            score += 15
        else:
            score += 5

    cls_conf = float(classification_dict.get("confidence", 0.0) or 0.0)
    if cls_conf < 0.60:
        score += 15

    score = max(0, min(score, 100))

    if score >= 70:
        label = "HIGH"
    elif score >= 40:
        label = "MEDIUM"
    else:
        label = "LOW"

    mandatory_review = label in {"HIGH", "MEDIUM"} or bool(flags)

    summary = (
        f"Fallback risk synthesis generated from validation flags and classification confidence. "
        f"Total score={score}, label={label}."
    )

    rule_hits = [
        {
            "rule_id": flag.get("rule", ""),
            "severity": flag.get("severity", ""),
            "status": flag.get("status", ""),
            "detail": flag.get("detail", ""),
        }
        for flag in flags
    ]

    return {
        "case_id": case_id,
        "risk_score": score,
        "risk_label": label,
        "mandatory_review": mandatory_review,
        "summary": summary,
        "rule_hits": rule_hits,
        "generated_by": "fallback_rules_pipeline",
    }


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

        aggregated_text = _aggregate_ocr_text(ocr_result)

        if not aggregated_text.strip():
            classification = classifier.classify("", doc_id=doc_id)
        else:
            classification = classifier.classify(aggregated_text, doc_id=doc_id)

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
        raise HTTPException(status_code=500, detail=f"pipeline failed: {str(exc)}") from exc
    finally:
        _pipeline_lock.release()


@router.post("/full-process", status_code=status.HTTP_200_OK)
def full_process(
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
            detail="Another pipeline job is already running. Please wait and retry."
        )

    doc_id = new_id("doc")
    ext = Path(file.filename).suffix.lower()
    dst_dir = ensure_dir(RAW_DIR / case_id / document_type)
    dst_path = dst_dir / f"{doc_id}{ext}"

    try:
        save_upload(file, dst_path)
        logger.info("Starting FULL PROCESS for case_id=%s doc_id=%s", case_id, doc_id)

        ocr_result = ocr_pipeline.process_document(case_id, doc_id, str(dst_path))
        aggregated_text = _aggregate_ocr_text(ocr_result)

        if not aggregated_text.strip():
            classification = classifier.classify("", doc_id=doc_id)
        else:
            classification = classifier.classify(aggregated_text, doc_id=doc_id)

        classification_dict = classification.to_dict()
        predicted_label = classification_dict.get("doc_type", "unknown")
        predicted_label = _override_doc_type_from_text(predicted_label, aggregated_text)
        normalized_doc_type = predicted_label if predicted_label != "unknown" else document_type
        classification_dict["doc_type"] = predicted_label

        run_extraction_pipeline, run_validation_pipeline, run_rules_pipeline = _safe_import_full_process_helpers()

        extraction_result = {}
        validation_result = {}
        rules_result = {}

        # extraction
        if run_extraction_pipeline:
            try:
                extraction_result = run_extraction_pipeline(
                    case_id=case_id,
                    document_id=doc_id,
                    doc_type=normalized_doc_type,
                    ocr_base_dir="data/ocr",
                    source_doc=str(dst_path),
                ) or {}
            except Exception as exc:
                logger.warning("run_extraction_pipeline failed, using fallback: %s", exc)

        if not extraction_result:
            extraction_result = _extract_fallback_entities(
                aggregated_text, case_id, doc_id, normalized_doc_type
            )

        extraction_path = Path("data/extraction") / case_id / "entities.json"
        _write_json(extraction_path, extraction_result)

        # validation
        if run_validation_pipeline:
            try:
                validation_result = run_validation_pipeline(
                    case_id=case_id,
                    document_id=doc_id,
                    doc_type=normalized_doc_type,
                ) or {}
            except Exception as exc:
                logger.warning("run_validation_pipeline failed, using fallback: %s", exc)

        if not validation_result:
            validation_result = _build_fallback_validation(extraction_result)

        validation_path = Path("data/validation") / case_id / "validation_result.json"
        _write_json(validation_path, validation_result)

        # rules / risk
        if run_rules_pipeline:
            try:
                rules_result = run_rules_pipeline(
                    case_id=case_id,
                    document_id=doc_id,
                    doc_type=normalized_doc_type,
                ) or {}
            except Exception as exc:
                logger.warning("run_rules_pipeline failed, using fallback: %s", exc)

        if not rules_result or not isinstance(rules_result, dict):
            rules_result = _build_fallback_rules(case_id, validation_result, classification_dict)

        risk_path = Path("data/results/risk_scores") / f"{case_id}_risk.json"
        _write_json(risk_path, rules_result)

        response = {
            "case_id": case_id,
            "document_id": doc_id,
            "document_type_hint": document_type,
            "predicted_doc_type": predicted_label,
            "resolved_doc_type": normalized_doc_type,
            "stored_path": str(dst_path),
            "ocr": ocr_result,
            "aggregated_text": aggregated_text,
            "classification": classification_dict,
            "extraction_result": extraction_result,
            "validation_result": validation_result,
            "rules_result": rules_result,
            "risk_file": str(risk_path),
            "risk_score": rules_result.get("risk_score", "—"),
            "risk_label": rules_result.get("risk_label", "—"),
            "mandatory_review": rules_result.get("mandatory_review", False),
            "risk_summary": rules_result.get("summary", ""),
            "rule_hits": rules_result.get("rule_hits", []),
        }

        return response

    except Exception as exc:
        logger.exception("Full process pipeline failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail=f"full process failed: {str(exc)}") from exc
    finally:
        _pipeline_lock.release()