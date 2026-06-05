"""
Phase 8.1 — DVC Stage: data_ingest
Reads raw documents from data/raw/, validates file formats,
copies them to data/ocr_input/ ready for OCR stage.
"""

import os
import shutil
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [data_ingest] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OCR_INPUT_DIR = Path("data/ocr_input")
MANIFEST_PATH = Path("data/ocr_input/ingest_manifest.json")

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 50


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_file(file_path: Path) -> tuple[bool, str]:
    """Validate a single file before ingest."""
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported extension: {file_path.suffix}"
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)"
    return True, "OK"


def ingest_documents():
    """Main ingest function: validate + copy raw docs to ocr_input."""
    OCR_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        logger.warning(f"RAW_DIR {RAW_DIR} does not exist — creating placeholder.")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        # Write a placeholder so DVC stage is satisfiable
        (OCR_INPUT_DIR / ".gitkeep").touch()
        _write_manifest([])
        return

    raw_files = [
        f for f in RAW_DIR.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    ]
    logger.info(f"Found {len(raw_files)} raw file(s) in {RAW_DIR}")

    manifest = []
    skipped = 0

    for src_path in sorted(raw_files):
        valid, reason = validate_file(src_path)
        if not valid:
            logger.warning(f"  SKIP {src_path.name}: {reason}")
            skipped += 1
            continue

        # Preserve relative sub-folder structure
        rel = src_path.relative_to(RAW_DIR)
        dst_path = OCR_INPUT_DIR / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        sha = compute_sha256(src_path)
        shutil.copy2(src_path, dst_path)

        entry = {
            "original_path": str(src_path),
            "staged_path": str(dst_path),
            "filename": src_path.name,
            "extension": src_path.suffix.lower(),
            "size_bytes": src_path.stat().st_size,
            "sha256": sha,
            "ingested_at": datetime.now(timezone.utc).isoformat(), 
            "status": "ok",
        }
        manifest.append(entry)
        logger.info(f"  OK  {src_path.name}  →  {dst_path}")

    logger.info(f"Ingested {len(manifest)} file(s), skipped {skipped}.")
    _write_manifest(manifest)


def _write_manifest(manifest: list):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_files": len(manifest),
                "records": manifest,
            },
            f,
            indent=2,
        )
    logger.info(f"Manifest written → {MANIFEST_PATH}")


if __name__ == "__main__":
    ingest_documents()