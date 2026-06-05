import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import pytesseract

from src.core.models import OCRBlock, OCRPageResult, PageQuality
from .preprocess_service import PreprocessService
from .file_service import ensure_dir

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

TESSERACT_EXE = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class OcrService:
    """
    OCR service with lazy PaddleOCR loading and safe fallback.
    engine:
      - "paddleocr" -> try Paddle first, then fallback to tesseract
      - "tesseract" -> force pytesseract
      - "stub"      -> no OCR, placeholder output
    """

    def __init__(
        self,
        engine: str = "tesseract",
        language: str = "eng",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ):
        self.engine = engine
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.preprocess = PreprocessService()
        self.ocr = None

    def _get_paddle_ocr(self):
        if self.ocr is not None:
            return self.ocr

        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            logger.info("PaddleOCR initialized successfully.")
            return self.ocr
        except Exception as e:
            logger.exception(f"PaddleOCR initialization failed: {e}")
            return None

    def run_ocr(self, image_bgr):
        if self.engine == "paddleocr":
            paddle = self._get_paddle_ocr()
            if paddle is not None:
                try:
                    return list(paddle.predict(input=image_bgr))
                except Exception as e:
                    logger.exception(f"PaddleOCR prediction failed: {e}")

        if self.engine in {"tesseract", "paddleocr"}:
            return self._run_tesseract(image_bgr)

        return {
            "text": "",
            "confidence": 0.0,
            "engine": "stub",
            "blocks": [],
        }

    def _run_tesseract(self, image_bgr):
        try:
            pil_img = Image.fromarray(image_bgr[:, :, ::-1])
            text = pytesseract.image_to_string(pil_img, lang=self.language)
            return {
                "text": text,
                "confidence": 0.0,
                "engine": "tesseract",
                "blocks": [],
            }
        except pytesseract.pytesseract.TesseractNotFoundError:
            logger.warning("Tesseract executable not found. Returning empty OCR result.")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "tesseract_missing",
                "blocks": [],
                "missing_dependency": "tesseract_binary",
            }
        except Exception as e:
            logger.exception(f"Tesseract OCR failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "tesseract_error",
                "error": str(e),
                "blocks": [],
            }

    def parse_ocr_output(self, raw):
        blocks, texts, confs = [], [], []

        if raw is None:
            return blocks, "", 0.0

        if isinstance(raw, dict) and "text" in raw:
            text = str(raw.get("text", ""))
            conf = float(raw.get("confidence", 0.0))
            return [], text, conf

        items = []
        if isinstance(raw, list):
            items = raw
        elif hasattr(raw, "to_dict"):
            items = raw.to_dict()
        elif hasattr(raw, "__iter__"):
            try:
                items = list(raw)
            except Exception:
                items = []

        page = items[0] if isinstance(items, list) and len(items) == 1 else items

        if isinstance(page, list):
            for item in page:
                if not item:
                    continue

                bbox = item[0] if len(item) > 0 else []
                rec = item[1] if len(item) > 1 else None

                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    text, conf = rec[0], rec[1]
                elif isinstance(rec, dict):
                    text = rec.get("text", "")
                    conf = rec.get("score", rec.get("confidence", 0.0))
                else:
                    text, conf = str(rec) if rec is not None else "", 0.0

                try:
                    flat = [float(v) for pt in bbox for v in pt]
                except Exception:
                    flat = []

                blocks.append(
                    OCRBlock(
                        text=str(text),
                        confidence=float(conf),
                        bbox=flat,
                    )
                )
                texts.append(str(text))
                confs.append(float(conf))

        elif isinstance(page, dict):
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_boxes = page.get("rec_boxes", page.get("dt_polys", []))

            for idx, text in enumerate(rec_texts):
                conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                bbox = rec_boxes[idx] if idx < len(rec_boxes) else None

                if bbox is not None:
                    try:
                        flat = [float(v) for pt in bbox for v in pt]
                    except Exception:
                        flat = []
                else:
                    flat = []

                blocks.append(
                    OCRBlock(
                        text=str(text),
                        confidence=conf,
                        bbox=flat,
                    )
                )
                texts.append(str(text))
                confs.append(conf)

        full_text = "\n".join(texts)
        page_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return blocks, full_text, page_conf

    def build_result(
        self,
        case_id: str,
        document_id: str,
        page_index: int,
        source_file: str,
        shape,
        quality: dict,
        raw,
    ):
        blocks, text, page_conf = self.parse_ocr_output(raw)

        return OCRPageResult(
            case_id=case_id,
            document_id=document_id,
            page_index=page_index,
            source_file=source_file,
            image_width=int(shape[1]),
            image_height=int(shape[0]),
            quality=PageQuality(**quality),
            engine=self.engine,
            language=self.language,
            page_confidence=page_conf,
            text=text,
            blocks=blocks,
            raw_result={"raw_type": str(type(raw).__name__)},
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

    def save_result(self, path: str | Path, result: OCRPageResult):
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return str(path)

    def _load_pdf_pages(self, file_path: Path):
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise RuntimeError(
                "pdf2image is required for PDF OCR. Install with: pip install pdf2image"
            ) from e

        pages = convert_from_path(str(file_path), dpi=250)
        return [page.convert("RGB") for page in pages]

    def _pil_to_bgr(self, image: Image.Image):
        rgb = np.array(image.convert("RGB"))
        return rgb[:, :, ::-1]

    def process(self, file_path: str) -> dict:
        source_path = Path(file_path)
        suffix = source_path.suffix.lower()

        all_blocks = []
        all_texts = []
        confidences = []

        if suffix == ".pdf":
            images = self._load_pdf_pages(source_path)
            source_type = "pdf"
        else:
            images = [Image.open(source_path).convert("RGB")]
            source_type = "image"

        for image in images:
            bgr, quality = self.preprocess.enhance(image)
            raw = self.run_ocr(bgr)
            blocks, text, page_conf = self.parse_ocr_output(raw)

            all_blocks.extend([b.model_dump() for b in blocks])
            all_texts.append(text)
            confidences.append(page_conf)

        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        return {
            "text": "\n\n".join([t for t in all_texts if t]),
            "pages": len(images),
            "confidence": avg_conf,
            "engine": self.engine,
            "language": self.language,
            "source_type": source_type,
            "quality": quality if images else {},
            "blocks": all_blocks,
        }