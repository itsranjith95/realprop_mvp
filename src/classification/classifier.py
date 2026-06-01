"""
DocumentClassifier — main orchestrator for Phase 3B.

Order:
  1. Preprocess OCR text
  2. ML classifier (strict MVP: mother_deed / khata_certificate / other)
  3. Rule engine fallback or support
  4. Optional Ollama support
  5. Confidence merge
"""

import logging
import re
import time
from typing import Optional

from .rule_engine import RuleEngine
from .ollama_client import OllamaClient
from .confidence import ConfidenceScorer, ClassificationResult
from .ml_classifier import MLDocumentClassifier

logger = logging.getLogger(__name__)


class DocumentClassifier:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "mistral",
        ollama_timeout: int = 120,
        use_ollama: bool = True,
    ):
        self.rule_engine = RuleEngine()
        self.scorer = ConfidenceScorer()
        self.ml_classifier = MLDocumentClassifier()
        self.use_ollama = use_ollama

        if use_ollama:
            self.ollama = OllamaClient(
                base_url=ollama_base_url,
                model=ollama_model,
                timeout=ollama_timeout,
            )
            if not self.ollama.is_available():
                logger.warning(
                    "Ollama not reachable at %s — falling back to ML/rule only.",
                    ollama_base_url,
                )
                self.use_ollama = False
        else:
            self.ollama = None

    def classify(self, text: str, doc_id: Optional[str] = None) -> ClassificationResult:
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

        ml_result = self.ml_classifier.predict(clean_text) if self.ml_classifier.is_available() else None
        rule_result = self.rule_engine.classify(clean_text)

        if ml_result and ml_result.get("error") is None:
            ml_doc = ml_result.get("doc_type", "other")
            ml_conf = ml_result.get("confidence", 0.0)

            if ml_doc in {"mother_deed", "khata_certificate"} and ml_conf >= 0.80:
                elapsed = time.perf_counter() - start
                logger.info(
                    "doc_id=%s | FINAL type=%s conf=%.2f method=ml_classifier review=%s time=%.2fs",
                    doc_id, ml_doc, ml_conf, False, elapsed,
                )
                return ClassificationResult(
                    doc_type=ml_doc,
                    confidence=ml_conf,
                    method="ml_classifier",
                    matched_keywords=rule_result.get("matched_keywords", []),
                    reasoning="Primary sklearn TF-IDF classifier",
                    rule_score=rule_result.get("confidence", 0.0),
                    ollama_score=0.0,
                    needs_human_review=False,
                    all_scores=ml_result.get("all_scores", {}),
                )

            if ml_doc == "other":
                # map strict MVP "other" into existing system semantics
                rule_conf = rule_result.get("confidence", 0.0)
                if rule_conf >= 0.55:
                    pass
                else:
                    return ClassificationResult(
                        doc_type="unknown",
                        confidence=min(ml_conf, 0.49),
                        method="ml_classifier",
                        matched_keywords=[],
                        reasoning="Primary sklearn TF-IDF classifier predicted other",
                        rule_score=rule_conf,
                        ollama_score=0.0,
                        needs_human_review=True,
                        all_scores=ml_result.get("all_scores", {}),
                    )

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

        final = self.scorer.merge(rule_result, ollama_result)

        elapsed = time.perf_counter() - start
        logger.info(
            "doc_id=%s | FINAL type=%s conf=%.2f method=%s review=%s time=%.2fs",
            doc_id, final.doc_type, final.confidence,
            final.method, final.needs_human_review, elapsed,
        )

        return final

    def classify_batch(self, texts: list[dict]) -> list[dict]:
        results = []
        for item in texts:
            doc_id = item.get("doc_id", "unknown")
            text = item.get("text", "")
            result = self.classify(text, doc_id=doc_id)
            results.append({"doc_id": doc_id, **result.to_dict()})
        return results

    def _preprocess(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[|]{2,}", " ", text)
        text = re.sub(r"[-]{3,}", " ", text)
        return text