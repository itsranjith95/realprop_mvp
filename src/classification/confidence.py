"""
Confidence scoring and decision logic.
Combines rule engine + Ollama results into a final classification.
"""

from dataclasses import dataclass, field
from typing import Optional


VALID_DOC_TYPES = {
    "sale_deed", "mother_deed", "khata_certificate", "khata_extract",
    "encumbrance_certificate", "property_tax_receipt", "aadhar_card",
    "pan_card", "power_of_attorney", "release_deed", "gift_deed",
    "will", "building_plan", "occupancy_certificate", "unknown",
}

# If rule engine confidence >= this, skip Ollama entirely
RULE_ENGINE_SKIP_THRESHOLD = 0.75

# Final result is flagged for human review if below this
HUMAN_REVIEW_THRESHOLD = 0.55


@dataclass
class ClassificationResult:
    doc_type: str
    confidence: float
    method: str                        # "rule_engine" | "ollama" | "ensemble"
    matched_keywords: list[str] = field(default_factory=list)
    reasoning: str = ""
    rule_score: float = 0.0
    ollama_score: float = 0.0
    needs_human_review: bool = False
    all_scores: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "doc_type": self.doc_type,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "matched_keywords": self.matched_keywords,
            "reasoning": self.reasoning,
            "rule_score": round(self.rule_score, 4),
            "ollama_score": round(self.ollama_score, 4),
            "needs_human_review": self.needs_human_review,
            "all_scores": self.all_scores,
            "error": self.error,
        }


class ConfidenceScorer:
    """
    Merges rule engine and Ollama results using a weighted ensemble.
    Rule engine weight: 0.45
    Ollama weight: 0.55
    Agreement bonus: +0.10 when both methods agree on doc_type
    """

    RULE_WEIGHT = 0.45
    OLLAMA_WEIGHT = 0.55
    AGREEMENT_BONUS = 0.10

    def should_skip_ollama(self, rule_result: dict) -> bool:
        return rule_result.get("confidence", 0.0) >= RULE_ENGINE_SKIP_THRESHOLD

    def merge(
        self,
        rule_result: dict,
        ollama_result: Optional[dict] = None,
    ) -> ClassificationResult:
        """
        Produces a final ClassificationResult by merging available signals.
        If ollama_result is None (skipped or failed), uses rule engine only.
        """
        rule_doc = rule_result.get("doc_type", "unknown")
        rule_conf = rule_result.get("confidence", 0.0)

        # Case 1: Ollama not used
        if ollama_result is None:
            final_conf = rule_conf
            final_doc = rule_doc
            method = "rule_engine"
            reasoning = ""
            ollama_conf = 0.0
        else:
            ollama_doc = ollama_result.get("doc_type", "unknown")
            ollama_conf = ollama_result.get("confidence", 0.0)
            ollama_error = ollama_result.get("error")

            # Case 2: Ollama errored — fall back to rule engine
            if ollama_error:
                final_conf = rule_conf
                final_doc = rule_doc
                method = "rule_engine"
                reasoning = f"Ollama error: {ollama_error}"
                ollama_conf = 0.0
            else:
                # Case 3: Full ensemble
                rule_weighted = rule_conf * self.RULE_WEIGHT
                ollama_weighted = ollama_conf * self.OLLAMA_WEIGHT
                base_conf = rule_weighted + ollama_weighted

                # Agreement bonus
                if rule_doc == ollama_doc and rule_doc != "unknown":
                    final_conf = min(1.0, base_conf + self.AGREEMENT_BONUS)
                    final_doc = rule_doc
                else:
                    # Disagreement — pick higher confidence, penalise
                    final_conf = max(rule_conf, ollama_conf) * 0.85
                    final_doc = (
                        rule_doc if rule_conf >= ollama_conf else ollama_doc
                    )

                method = "ensemble"
                reasoning = ollama_result.get("reasoning", "")

        # Validate doc type
        if final_doc not in VALID_DOC_TYPES:
            final_doc = "unknown"
            final_conf = 0.0
            
        if final_doc == "unknown":
            final_conf = min(final_conf, 0.49)

        return ClassificationResult(
            doc_type=final_doc,
            confidence=round(final_conf, 4),
            method=method,
            matched_keywords=rule_result.get("matched_keywords", []),
            reasoning=reasoning,
            rule_score=rule_conf,
            ollama_score=ollama_conf,
            needs_human_review=(final_doc == "unknown" or final_conf < HUMAN_REVIEW_THRESHOLD),
            all_scores=rule_result.get("all_scores", {}),
        )