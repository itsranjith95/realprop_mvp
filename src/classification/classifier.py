"""
DocumentClassifier — main orchestrator for Phase 3.
Pipeline:
  1. Preprocess / clean OCR text
  2. Rule engine pass
  3. Decide if Ollama is needed
  4. Ollama pass (if needed)
  5. Ensemble merge + confidence scoring
  6. Return ClassificationResult
"""

import logging
import re
import time
from typing import Optional

from .rule_engine import RuleEngine
from .ollama_client import OllamaClient
from .confidence import ConfidenceScorer, ClassificationResult

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Main classifier. Instantiate once and reuse across requests.

    Usage:
        classifier = DocumentClassifier()
        result = classifier.classify(ocr_text)
        print(result.doc_type, result.confidence)
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "mistral",
        ollama_timeout: int = 120,
        use_ollama: bool = True,
    ):
        self.rule_engine = RuleEngine()
        self.scorer = ConfidenceScorer()
        self.use_ollama = use_ollama

        if use_ollama:
            self.ollama = OllamaClient(
                base_url=ollama_base_url,
                model=ollama_model,
                timeout=ollama_timeout,
            )
            if not self.ollama.is_available():
                logger.warning(
                    "Ollama not reachable at %s — falling back to rule engine only.",
                    ollama_base_url,
                )
                self.use_ollama = False
        else:
            self.ollama = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str, doc_id: Optional[str] = None) -> ClassificationResult:
        """
        Classify a document from its OCR/extracted text.

        Args:
            text: Full text extracted from the document (Phase 2 output).
            doc_id: Optional document identifier for logging.

        Returns:
            ClassificationResult dataclass.
        """
        start = time.perf_counter()

        if not text or not text.strip():
            logger.warning("Empty text received for doc_id=%s", doc_id)
            return ClassificationResult(
                doc_type="unknown",
                confidence=0.0,
                method="rule_engine",
                needs_human_review=True,
                error="empty_text",
            )

        clean_text = self._preprocess(text)

        # Step 1: Rule engine
        rule_result = self.rule_engine.classify(clean_text)
        logger.debug(
            "doc_id=%s | rule=%s conf=%.2f",
            doc_id, rule_result["doc_type"], rule_result["confidence"],
        )

        # Step 2: Decide if Ollama is needed
        ollama_result = None
        if self.use_ollama and self.ollama:
            if self.scorer.should_skip_ollama(rule_result):
                logger.debug(
                    "doc_id=%s | Skipping Ollama (rule confidence %.2f >= threshold)",
                    doc_id, rule_result["confidence"],
                )
            else:
                logger.debug("doc_id=%s | Calling Ollama model=%s", doc_id, self.ollama.model)
                ollama_result = self.ollama.classify(clean_text)
                logger.debug(
                    "doc_id=%s | ollama=%s conf=%.2f",
                    doc_id,
                    ollama_result.get("doc_type"),
                    ollama_result.get("confidence", 0.0),
                )

        # Step 3: Merge results
        final = self.scorer.merge(rule_result, ollama_result)

        elapsed = time.perf_counter() - start
        logger.info(
            "doc_id=%s | FINAL type=%s conf=%.2f method=%s review=%s time=%.2fs",
            doc_id, final.doc_type, final.confidence,
            final.method, final.needs_human_review, elapsed,
        )

        return final

    def classify_batch(self, texts: list[dict]) -> list[dict]:
        """
        Classify multiple documents.
        texts: list of {"doc_id": str, "text": str}
        Returns: list of {"doc_id": str, **ClassificationResult.to_dict()}
        """
        results = []
        for item in texts:
            doc_id = item.get("doc_id", "unknown")
            text = item.get("text", "")
            result = self.classify(text, doc_id=doc_id)
            results.append({"doc_id": doc_id, **result.to_dict()})
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """Clean OCR noise before classification."""
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove null bytes / control chars
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse repeated punctuation (OCR artefacts)
        text = re.sub(r"[|]{2,}", " ", text)
        text = re.sub(r"[-]{3,}", " ", text)
        return text