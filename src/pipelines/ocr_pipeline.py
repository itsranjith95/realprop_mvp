"""
Phase 8.1 — DVC Stage: ocr
Reads staged files from data/ocr_input/, runs OCR pipeline,
writes per-document/page OCR JSON outputs under data/ocr/
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ocr_pipeline] %(message)s")
logger = logging.getLogger(__name__)

OCR_INPUT_DIR = Path("data/ocr_input")
OCR_OUTPUT_DIR = Path("data/ocr")


def _get_ocr_pipeline():
    try:
        from src.services.ocr_pipeline import OCRPipeline
        return OCRPipeline(ocr_dir=str(OCR_OUTPUT_DIR))
    except Exception as e:
        logger.exception(f"OCRPipeline import failed: {e}")
        return None


def derive_ids(file_path: Path) -> tuple[str, str]:
    """
    Example:
      data/ocr_input/khata/Anithas_khata.png
      data/ocr_input/motherdeed/bangalore_sample_mother_deed.pdf

    case_id => first folder below ocr_input
    doc_id  => filename stem
    """
    rel = file_path.relative_to(OCR_INPUT_DIR)
    parts = rel.parts

    if len(parts) >= 2:
        case_id = parts[0]
        doc_id = Path(parts[-1]).stem
    else:
        case_id = "default_case"
        doc_id = file_path.stem

    return case_id, doc_id


def run_ocr_pipeline():
    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ocr_pipeline = _get_ocr_pipeline()

    if not OCR_INPUT_DIR.exists():
        logger.warning("ocr_input dir missing — nothing to OCR.")
        (OCR_OUTPUT_DIR / ".gitkeep").touch()
        return

    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    files = [
        f for f in OCR_INPUT_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    ]

    logger.info(f"Running OCR on {len(files)} file(s).")

    if ocr_pipeline is None:
        logger.error("OCRPipeline could not be loaded. Aborting OCR stage.")
        raise RuntimeError("OCRPipeline import failed")

    for file_path in sorted(files):
        case_id, doc_id = derive_ids(file_path)
        logger.info(f"  OCR  {file_path.name}  →  data/ocr/{case_id}/{doc_id}/")

        try:
            ocr_pipeline.process_document(
                case_id=case_id,
                document_id=doc_id,
                source_path=str(file_path),
            )
        except Exception as e:
            logger.exception(f"OCR failed for {file_path}: {e}")
            raise

    logger.info("OCR pipeline complete.")


if __name__ == "__main__":
    run_ocr_pipeline()