# src/pipelines/classification_pipeline.py
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from src.classification.ml_classifier import MLDocumentClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [classification_pipeline] %(message)s")
logger = logging.getLogger(__name__)

OCR_DIR = Path("data/ocr")
CLASSIFICATION_DIR = Path("data/classification")
MODEL_PATH = Path("data/models/classification/tfidf_logreg.joblib")


def extract_text_from_manifest(manifest_path: Path) -> str:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        pages = data.get("pages", [])
        texts = []

        for page in pages:
            if isinstance(page, dict):
                if "full_text" in page and page["full_text"]:
                    texts.append(str(page["full_text"]))
                elif "text" in page and page["text"]:
                    texts.append(str(page["text"]))
                elif "ocr_text" in page and page["ocr_text"]:
                    texts.append(str(page["ocr_text"]))

        return "\n".join(texts).strip()
    except Exception as e:
        logger.warning(f"Failed to extract text from {manifest_path}: {e}")
        return ""


def run_classification_pipeline():
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    classifier = MLDocumentClassifier(MODEL_PATH)
    if not classifier.is_available():
        raise RuntimeError(f"Classifier model not available at {MODEL_PATH}")

    manifest_files = sorted(OCR_DIR.rglob("manifest.json"))
    logger.info(f"Found {len(manifest_files)} OCR manifest file(s).")

    for manifest_path in manifest_files:
        rel = manifest_path.relative_to(OCR_DIR)
        case_id = rel.parts[0] if len(rel.parts) > 0 else "default_case"
        doc_id = rel.parts[1] if len(rel.parts) > 1 else manifest_path.parent.name

        text = extract_text_from_manifest(manifest_path)
        prediction = classifier.predict(text)

        out_dir = CLASSIFICATION_DIR / case_id / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "case_id": case_id,
            "doc_id": doc_id,
            "source_manifest": str(manifest_path),
            "classified_at": datetime.now(timezone.utc).isoformat(),
            "input_text_chars": len(text),
            "prediction": prediction,
        }

        out_path = out_dir / "prediction.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Predicted {case_id}/{doc_id} → {prediction['doc_type']} ({prediction['confidence']})")

    logger.info("Classification pipeline complete.")


if __name__ == "__main__":
    run_classification_pipeline()