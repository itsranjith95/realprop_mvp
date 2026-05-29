"""
Rule-based document classifier for Indian property documents.
Runs BEFORE Ollama — if confidence is high enough, Ollama is skipped.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Keyword maps — order matters (more specific first)
# ---------------------------------------------------------------------------

DOCUMENT_RULES: dict[str, list[str]] = {
    "sale_deed": [
        "sale deed", "absolute sale", "vendor", "vendee",
        "conveyance deed", "transfer of property", "sale consideration",
        "registered sale", "sub-registrar", "stamp duty",
    ],
    "mother_deed": [
        "mother deed", "parent document", "original deed",
        "chain of title", "prior deed",
    ],
    "khata_certificate": [
        "khata certificate", "khata no", "khatha certificate",
        "bbmp", "bruhat bengaluru", "municipal corporation",
        "property tax assessment", "assessment number",
    ],
    "khata_extract": [
        "khata extract", "khatha extract",
        "assessment register", "ward no", "zone no",
        "khata transfer",
    ],
    "encumbrance_certificate": [
        "encumbrance certificate", "ec certificate",
        "sub-registrar office", "form 15", "form 16",
        "no encumbrance", "encumbrance details",
        "transactions registered",
    ],
    "property_tax_receipt": [
        "property tax receipt", "tax paid receipt",
        "tax payment", "demand notice", "tax bill",
        "arrears", "current tax", "bbmp tax",
        "property identification number", "pid",
    ],
    "aadhar_card": [
        "aadhaar", "aadhar", "unique identification",
        "uidai", "enrolment no", "vid:",
        "government of india", "dob:", "male", "female",
    ],
    "pan_card": [
        "permanent account number", "pan card",
        "income tax department", "govt. of india",
        "father's name",
    ],
    "power_of_attorney": [
        "power of attorney", "poa", "general power",
        "special power", "attorney", "principal",
        "authorize", "authorise",
    ],
    "release_deed": [
        "release deed", "relinquishment deed",
        "co-owner", "relinquish", "release his share",
        "partition", "family settlement",
    ],
    "gift_deed": [
        "gift deed", "donor", "donee",
        "voluntary transfer", "natural love",
        "without consideration",
    ],
    "will": [
        "last will", "testament", "testator",
        "bequeath", "legatee", "executor",
        "codicil",
    ],
    "building_plan": [
        "building plan", "approved plan", "site plan",
        "floor plan", "bda", "bbmp approval",
        "rera", "layout plan", "sanctioned plan",
    ],
    "occupancy_certificate": [
        "occupancy certificate", "completion certificate",
        "oc certificate", "fit for occupation",
        "bwssb", "bescom",
    ],
}

CONFIDENCE_THRESHOLDS = {
    "high": 0.80,
    "medium": 0.55,
    "low": 0.30,
}


class RuleEngine:
    """
    Heuristic keyword-based classifier.
    Returns (doc_type, confidence, matched_keywords).
    """

    def __init__(self):
        self._compiled = self._compile_rules()

    def _compile_rules(self) -> dict[str, list[re.Pattern]]:
        compiled = {}
        for doc_type, keywords in DOCUMENT_RULES.items():
            compiled[doc_type] = [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in keywords
            ]
        return compiled

    def classify(self, text: str) -> dict:
        """
        Returns:
            {
                "doc_type": str,
                "confidence": float,
                "matched_keywords": list[str],
                "all_scores": dict[str, float],
                "method": "rule_engine"
            }
        """
        text_lower = text.lower()
        scores: dict[str, float] = {}
        matched_map: dict[str, list[str]] = {}

        for doc_type, patterns in self._compiled.items():
            hits = []
            for pattern in patterns:
                if pattern.search(text_lower):
                    hits.append(pattern.pattern.replace(r"\b", "").replace(r"\ ", " "))
            if hits:
                total_keywords = len(DOCUMENT_RULES[doc_type])
                raw_score = len(hits) / total_keywords
                # bonus for multiple hits — diminishing returns
                bonus = min(0.20, (len(hits) - 1) * 0.05)
                scores[doc_type] = min(1.0, raw_score + bonus)
                matched_map[doc_type] = hits

        if not scores:
            return {
                "doc_type": "unknown",
                "confidence": 0.0,
                "matched_keywords": [],
                "all_scores": {},
                "method": "rule_engine",
            }

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Penalise if second-best is close (ambiguous doc)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            if gap < 0.10:
                best_score = max(0.0, best_score - 0.15)

        return {
            "doc_type": best_type,
            "confidence": round(best_score, 4),
            "matched_keywords": matched_map.get(best_type, []),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
            "method": "rule_engine",
        }