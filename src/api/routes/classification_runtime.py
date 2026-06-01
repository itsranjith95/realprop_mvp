import csv
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.classification import DocumentClassifier
from src.services.file_service import ensure_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/runtime", tags=["runtime-classification"])

classifier = DocumentClassifier()

LABELED_DIR = Path("data/labeled/classification")
TRAIN_PATH = LABELED_DIR / "train.csv"


class ClassifyTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    doc_id: str | None = None


class ClassifyManifestRequest(BaseModel):
    manifest_path: str


class SaveReviewRequest(BaseModel):
    case_id: str
    document_id: str
    confirmed_label: str
    aggregated_text: str
    source: str = "manual_review"
    review_notes: str = ""


def _extract_text_from_page(page: dict[str, Any]) -> list[str]:
    chunks: list[str] = []

    if "text" in page and isinstance(page["text"], str):
        chunks.append(page["text"])

    if "raw_text" in page and isinstance(page["raw_text"], str):
        chunks.append(page["raw_text"])

    if "blocks" in page and isinstance(page["blocks"], list):
        for block in page["blocks"]:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    chunks.append(str(text))

    if "lines" in page and isinstance(page["lines"], list):
        for line in page["lines"]:
            if isinstance(line, dict):
                text = line.get("text")
                if text:
                    chunks.append(str(text))
            elif isinstance(line, str):
                chunks.append(line)

    return chunks


def aggregate_manifest_text(manifest_data: dict[str, Any]) -> str:
    pages = manifest_data.get("pages", [])
    all_chunks: list[str] = []

    for page in pages:
        if isinstance(page, dict):
            all_chunks.extend(_extract_text_from_page(page))

    return "\n".join(chunk.strip() for chunk in all_chunks if chunk and chunk.strip()).strip()


@router.post("/classify-text")
def classify_text(req: ClassifyTextRequest):
    result = classifier.classify(req.text, doc_id=req.doc_id)
    return {
        "doc_id": req.doc_id,
        "aggregated_text": req.text,
        "classification": result.to_dict(),
    }


@router.post("/classify-manifest")
def classify_manifest(req: ClassifyManifestRequest):
    manifest_path = Path(req.manifest_path)

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest file not found")

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid manifest JSON: {exc}") from exc

    aggregated_text = aggregate_manifest_text(manifest_data)
    document_id = manifest_data.get("document_id")
    case_id = manifest_data.get("case_id")

    result = classifier.classify(aggregated_text, doc_id=document_id)

    return {
        "manifest_path": str(manifest_path),
        "document_id": document_id,
        "case_id": case_id,
        "aggregated_text": aggregated_text,
        "classification": result.to_dict(),
        "ocr": manifest_data,
    }


@router.post("/save-review-example")
def save_review_example(req: SaveReviewRequest):
    allowed_labels = {"mother_deed", "khata_certificate", "other"}
    if req.confirmed_label not in allowed_labels:
        raise HTTPException(
            status_code=400,
            detail=f"confirmed_label must be one of: {sorted(allowed_labels)}",
        )

    if not req.aggregated_text.strip():
        raise HTTPException(status_code=400, detail="aggregated_text is required")

    ensure_dir(LABELED_DIR)

    file_exists = TRAIN_PATH.exists()
    with TRAIN_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "label", "case_id", "document_id", "source", "review_notes"],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "text": req.aggregated_text.strip(),
                "label": req.confirmed_label,
                "case_id": req.case_id,
                "document_id": req.document_id,
                "source": req.source,
                "review_notes": req.review_notes,
            }
        )

    return {
        "status": "saved",
        "target_file": str(TRAIN_PATH),
        "document_id": req.document_id,
        "label": req.confirmed_label,
    }