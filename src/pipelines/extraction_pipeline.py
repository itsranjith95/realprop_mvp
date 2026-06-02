"""
extraction_pipeline.py
----------------------
Phase 4 – Entity Extraction and Normalisation Pipeline.

Reads OCR JSON from data/ocr/<case_id>/<doc_id>/page*.json,
runs extraction + normalisation, persists to SQLite, logs to MLflow.

DVC stage : extraction
CLI       : python -m src.pipelines.extraction_pipeline \
                --case_id <uuid> --doc_id <uuid> --doc_type motherdeed|khata
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow

from src.core.database import SessionLocal
from src.core.models import ExtractedEntity, OCRBlock, OCRPageResult
from src.services.extractionservice import extract_entities, persist_entities

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def _load_ocr_pages(ocr_dir: Path) -> list[OCRPageResult]:
    pages = []
    for json_file in sorted(ocr_dir.glob("page*.json")):
        with open(json_file) as f:
            pages.append(OCRPageResult(**json.load(f)))
    return pages


def run_extraction_pipeline(
    case_id: str,
    document_id: str,
    doc_type: str,
    ocr_base_dir: str = "data/ocr",
    source_doc: str = "",
) -> list[ExtractedEntity]:
    """
    Steps:
    1. Load OCR page JSON files
    2. Extract + normalise entities per page
    3. Persist all to SQLite
    4. Log metrics to MLflow
    """
    ocr_dir = Path(ocr_base_dir) / case_id / document_id
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR directory not found: {ocr_dir}")

    pages = _load_ocr_pages(ocr_dir)
    if not pages:
        logger.warning("No OCR page files in %s", ocr_dir)
        return []

    all_entities: list[ExtractedEntity] = []

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("extraction_pipeline")

    with mlflow.start_run(run_name=f"{doc_type}_{document_id[:8]}"):
        mlflow.log_params({"case_id": case_id, "document_id": document_id,
                           "doc_type": doc_type, "total_pages": len(pages)})

        for page in pages:
            entities = extract_entities(doc_type, page.blocks, page.page_index)
            all_entities.extend(entities)

        db = SessionLocal()
        try:
            persist_entities(case_id, document_id, source_doc, all_entities, db)
        finally:
            db.close()

        avg_conf = sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0.0
        field_counts: dict[str, int] = {}
        for e in all_entities:
            field_counts[e.field_name] = field_counts.get(e.field_name, 0) + 1

        mlflow.log_metric("total_entities_extracted", len(all_entities))
        mlflow.log_metric("avg_confidence", round(avg_conf, 4))
        mlflow.log_metric("unique_fields", len(field_counts))
        for field, count in field_counts.items():
            mlflow.log_metric(f"field_{field}_count", count)

        logger.info("Extraction complete: %d entities, avg_conf=%.3f", len(all_entities), avg_conf)

    return all_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id",      required=True)
    parser.add_argument("--doc_id",       required=True)
    parser.add_argument("--doc_type",     required=True, choices=["motherdeed","khata"])
    parser.add_argument("--ocr_base_dir", default="data/ocr")
    parser.add_argument("--source_doc",   default="")
    args = parser.parse_args()

    entities = run_extraction_pipeline(
        case_id=args.case_id, document_id=args.doc_id,
        doc_type=args.doc_type, ocr_base_dir=args.ocr_base_dir,
        source_doc=args.source_doc,
    )
    print(f"Extracted {len(entities)} entities.")
    for e in entities:
        print(f"  [{e.field_name}] raw='{e.value}' | norm='{e.normalized_value}' | conf={e.confidence:.2f}")