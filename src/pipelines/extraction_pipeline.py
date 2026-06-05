"""
Phase 8.1 — Extraction Pipeline
Reads OCR page JSON from data/ocr/<case_id>/<doc_id>/page*.json,
reads predicted doc type from data/classification/<case_id>/<doc_id>/prediction.json,
runs entity extraction + normalization, persists to SQLite,
logs metrics to MLflow, and writes JSON outputs to data/extracted/.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

import mlflow

from src.core.database import SessionLocal
from src.core.models import ExtractedEntity, OCRBlock, OCRPageResult
from src.services.extractionservice import extract_entities, persist_entities

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [extraction_pipeline] %(message)s")

OCR_BASE_DIR = Path("data/ocr")
CLASSIFICATION_BASE_DIR = Path("data/classification")
EXTRACTED_BASE_DIR = Path("data/extracted")


def _load_ocr_pages(ocr_dir: Path) -> list[OCRPageResult]:
    pages = []
    for json_file in sorted(ocr_dir.glob("page*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            pages.append(OCRPageResult(**json.load(f)))
    return pages


def _load_prediction(prediction_path: Path) -> dict:
    with open(prediction_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_doc_type(raw_doc_type: str) -> str:
    doc_type = (raw_doc_type or "").strip().lower()
    mapping = {
        "mother_deed": "motherdeed",
        "motherdeed": "motherdeed",
        "khata_certificate": "khata",
        "khata": "khata",
    }
    return mapping.get(doc_type, "other")


def _page_to_blocks(page: OCRPageResult) -> list[OCRBlock]:
    if page.blocks:
        return page.blocks

    fallback_text = (page.text or "").strip()
    if not fallback_text:
        return []

    logger.warning(
        f"Page {page.page_index} for {page.document_id} has empty blocks; "
        f"falling back to synthetic block from page.text"
    )

    return [
        OCRBlock(
            text=fallback_text,
            confidence=max(0.0, min(1.0, float(page.page_confidence or 0.75))),
            bbox=[0.0, 0.0, float(page.image_width), float(page.image_height)],
            block_type="page_text_fallback",
        )
    ]


def _entity_to_dict(entity: ExtractedEntity) -> dict:
    if hasattr(entity, "model_dump"):
        return entity.model_dump()
    if hasattr(entity, "dict"):
        return entity.dict()
    return {
        "field_name": getattr(entity, "field_name", None),
        "value": getattr(entity, "value", None),
        "normalized_value": getattr(entity, "normalized_value", None),
        "confidence": getattr(entity, "confidence", None),
        "page": getattr(entity, "page", None),
        "bbox": getattr(entity, "bbox", []),
        "source_doc": getattr(entity, "source_doc", ""),
        "extraction_method": getattr(entity, "extraction_method", "regex"),
        "created_at": getattr(entity, "created_at", ""),
    }


def run_extraction_for_document(
    case_id: str,
    document_id: str,
    doc_type: str,
    ocr_base_dir: Path = OCR_BASE_DIR,
    source_doc: str = "",
) -> list[ExtractedEntity]:
    ocr_dir = ocr_base_dir / case_id / document_id
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR directory not found: {ocr_dir}")

    pages = _load_ocr_pages(ocr_dir)
    if not pages:
        logger.warning(f"No OCR page files found in {ocr_dir}")
        return []

    all_entities: list[ExtractedEntity] = []

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("extraction_pipeline")

    with mlflow.start_run(run_name=f"{doc_type}_{document_id[:32]}"):
        mlflow.log_params({
            "case_id": case_id,
            "document_id": document_id,
            "doc_type": doc_type,
            "total_pages": len(pages),
        })

        for page in pages:
            blocks = _page_to_blocks(page)
            entities = extract_entities(doc_type, blocks, page.page_index)
            all_entities.extend(entities)

        db = SessionLocal()
        try:
            persist_entities(case_id, document_id, source_doc, all_entities, db)
        finally:
            db.close()

        avg_conf = (
            sum(float(e.confidence) for e in all_entities) / len(all_entities)
            if all_entities else 0.0
        )

        field_counts: dict[str, int] = {}
        for e in all_entities:
            field_name = str(e.field_name)
            field_counts[field_name] = field_counts.get(field_name, 0) + 1

        mlflow.log_metric("total_entities_extracted", len(all_entities))
        mlflow.log_metric("avg_confidence", round(avg_conf, 4))
        mlflow.log_metric("unique_fields", len(field_counts))

        for field, count in field_counts.items():
            mlflow.log_metric(f"field_{field}_count", count)

        logger.info(
            f"Extraction complete for {case_id}/{document_id}: "
            f"{len(all_entities)} entities, avg_conf={avg_conf:.3f}"
        )

    return all_entities


def run_extraction_pipeline():
    EXTRACTED_BASE_DIR.mkdir(parents=True, exist_ok=True)

    prediction_files = sorted(CLASSIFICATION_BASE_DIR.rglob("prediction.json"))
    logger.info(f"Found {len(prediction_files)} classification prediction file(s).")

    processed_docs = 0

    for prediction_path in prediction_files:
        rel = prediction_path.relative_to(CLASSIFICATION_BASE_DIR)
        if len(rel.parts) < 3:
            logger.warning(f"Skipping malformed prediction path: {prediction_path}")
            continue

        case_id = rel.parts[0]
        document_id = rel.parts[1]

        try:
            prediction_payload = _load_prediction(prediction_path)
            prediction = prediction_payload.get("prediction", {})
            raw_doc_type = prediction.get("doc_type", "other")
            confidence = prediction.get("confidence", 0.0)
            doc_type = _normalize_doc_type(raw_doc_type)

            if doc_type == "other":
                logger.warning(
                    f"Skipping {case_id}/{document_id}: unsupported predicted doc_type={raw_doc_type}"
                )
                continue

            entities = run_extraction_for_document(
                case_id=case_id,
                document_id=document_id,
                doc_type=doc_type,
                ocr_base_dir=OCR_BASE_DIR,
                source_doc=str(OCR_BASE_DIR / case_id / document_id),
            )

            out_dir = EXTRACTED_BASE_DIR / case_id / document_id
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / "entities.json"
            payload = {
                "case_id": case_id,
                "document_id": document_id,
                "doc_type": doc_type,
                "classification_confidence": confidence,
                "source_prediction": str(prediction_path),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "entity_count": len(entities),
                "entities": [_entity_to_dict(e) for e in entities],
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved extraction output → {out_path} "
                f"({len(entities)} entities, doc_type={doc_type})"
            )
            processed_docs += 1

        except Exception as e:
            logger.exception(f"Extraction failed for {prediction_path}: {e}")
            raise

    logger.info(f"Extraction pipeline complete. Processed {processed_docs} document(s).")


if __name__ == "__main__":
    run_extraction_pipeline()
    