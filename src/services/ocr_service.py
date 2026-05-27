import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest import result

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR
from src.core.models import OCRBlock, OCRPageResult, PageQuality
from .preprocess_service import PreprocessService
from .file_service import ensure_dir

logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)


class OCRService:
    def __init__(
        self,
        engine: str = "paddleocr",
        language: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ): 
        self.engine = engine
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.ocr = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        self.preprocess = PreprocessService()

    def run_ocr(self, image_bgr):
        print("DEBUG IMAGE SHAPE:", getattr(image_bgr, "shape", None))
        result = list(self.ocr.predict(input=image_bgr))
        print("DEBUG PADDLEOCR RAW:", result)
        return result

    def parse_ocr_output(self, raw):
        blocks, texts, confs = [], [], []

        if raw is None:
            return blocks, "", 0.0

        # Normalize raw into "page"
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

        # Case 1: old-style PaddleOCR output: list of [bbox, (text, conf)]
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

                flat = []
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

        # Case 2: new-style dict from self.ocr.predict(...)
        elif isinstance(page, dict):
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_boxes = page.get("rec_boxes", page.get("dt_polys", []))

            for idx, text in enumerate(rec_texts):
                conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                bbox = rec_boxes[idx] if idx < len(rec_boxes) else None

                flat = []
                if bbox is not None:
                    try:
                        flat = [float(v) for pt in bbox for v in pt]
                    except Exception:
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