import json
from pathlib import Path
from PIL import Image

from .file_service import ensure_dir
from .pdf_service import pdf_to_images
from .ocr_service import OcrService


class OCRPipeline:
    """
    Internal OCR pipeline helper that processes a single document
    into page-wise OCR JSON files plus a manifest.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        interim_dir: str = "data/interim",
        ocr_dir: str = "data/ocr",
        dpi: int = 300,
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.ocr_dir = Path(ocr_dir)
        self.dpi = dpi
        self.service = OcrService()

    def _load_pages(self, file_path: Path):
        if file_path.suffix.lower() == ".pdf":
            return pdf_to_images(str(file_path), dpi=self.dpi)
        return [(0, Image.open(file_path).convert("RGB"))]

    def process_document(self, case_id: str, document_id: str, source_path: str):
        source_path = Path(source_path)
        pages = self._load_pages(source_path)
        results = []

        for page_index, page_img in pages:
            bgr, quality = self.service.preprocess.enhance(page_img)
            raw = self.service.run_ocr(bgr)
            result = self.service.build_result(
                case_id=case_id,
                document_id=document_id,
                page_index=page_index,
                source_file=str(source_path),
                shape=bgr.shape,
                quality=quality,
                raw=raw,
            )

            out_dir = ensure_dir(self.ocr_dir / case_id / document_id)
            out_file = out_dir / f"page_{page_index:03d}.json"
            self.service.save_result(out_file, result)
            results.append(result.model_dump())

        manifest = self.ocr_dir / case_id / document_id / "manifest.json"
        ensure_dir(manifest.parent)
        manifest.write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "document_id": document_id,
                    "pages": results,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return {
            "status": "OCR_DONE",
            "manifest": str(manifest),
            "pages": results,
        }