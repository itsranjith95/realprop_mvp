"""
Rule-based document classifier for Indian property documents.
Strict MVP focus: mother_deed and khata_certificate.
Other labels remain supported for backward compatibility.
"""

import re

DOCUMENT_RULES: dict[str, list[str]] = {
    "mother_deed": [
        "mother deed",
        "parent document",
        "original deed",
        "chain of title",
        "prior deed",
        "schedule of property",
        "schedule property",
        "executant",
        "claimant",
        "absolute owner",
        "sale consideration",
        "sub-registrar",
        "registered as document",
        "book i",
        "document no",
        "site measuring",
        "all that piece and parcel",
        "property bearing",
        "bounded on the",
    ],
    "khata_certificate": [
        "khata certificate",
        "khatha certificate",
        "khata no",
        "khatha no",
        "bbmp",
        "bruhat bengaluru mahanagara palike",
        "bengaluru mahanagara palike",
        "municipal corporation",
        "assessment number",
        "property tax assessment",
        "pid",
        "sas application",
        "ward",
        "zone",
        "owner name",
        "khata transfer",
        "greater bengaluru authority",
    ],
    "sale_deed": [
        "sale deed",
        "absolute sale",
        "vendor",
        "vendee",
        "conveyance deed",
        "transfer of property",
        "sale consideration",
        "registered sale",
        "sub-registrar",
        "stamp duty",
    ],
    "khata_extract": [
        "khata extract",
        "khatha extract",
        "assessment register",
        "ward no",
        "zone no",
        "khata transfer",
    ],
    "encumbrance_certificate": [
        "encumbrance certificate",
        "ec certificate",
        "sub-registrar office",
        "form 15",
        "form 16",
        "no encumbrance",
        "encumbrance details",
        "transactions registered",
    ],
    "property_tax_receipt": [
        "property tax receipt",
        "tax paid receipt",
        "tax payment",
        "demand notice",
        "tax bill",
        "arrears",
        "current tax",
        "bbmp tax",
        "property identification number",
        "pid",
    ],
    "aadhar_card": [
        "aadhaar",
        "aadhar",
        "unique identification",
        "uidai",
        "enrolment no",
        "vid:",
        "government of india",
        "dob:",
        "male",
        "female",
    ],
    "pan_card": [
        "permanent account number",
        "pan card",
        "income tax department",
        "govt. of india",
        "father's name",
    ],
    "power_of_attorney": [
        "power of attorney",
        "poa",
        "general power",
        "special power",
        "attorney",
        "principal",
        "authorize",
        "authorise",
    ],
    "release_deed": [
        "release deed",
        "relinquishment deed",
        "co-owner",
        "relinquish",
        "release his share",
        "partition",
        "family settlement",
    ],
    "gift_deed": [
        "gift deed",
        "donor",
        "donee",
        "voluntary transfer",
        "natural love",
        "without consideration",
    ],
    "will": [
        "last will",
        "testament",
        "testator",
        "bequeath",
        "legatee",
        "executor",
        "codicil",
    ],
    "building_plan": [
        "building plan",
        "approved plan",
        "site plan",
        "floor plan",
        "bda",
        "bbmp approval",
        "rera",
        "layout plan",
        "sanctioned plan",
    ],
    "occupancy_certificate": [
        "occupancy certificate",
        "completion certificate",
        "oc certificate",
        "fit for occupation",
        "bwssb",
        "bescom",
    ],
}

CONFIDENCE_THRESHOLDS = {
    "high": 0.80,
    "medium": 0.55,
    "low": 0.30,
}


class RuleEngine:
    def __init__(self):
        self._compiled = self._compile_rules()

    def _compile_rules(self) -> dict[str, list[tuple[str, re.Pattern]]]:
        compiled: dict[str, list[tuple[str, re.Pattern]]] = {}
        for doc_type, keywords in DOCUMENT_RULES.items():
            compiled[doc_type] = [
                (
                    kw,
                    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                )
                for kw in keywords
            ]
        return compiled

    def classify(self, text: str) -> dict:
        text_lower = text.lower()
        scores: dict[str, float] = {}
        matched_map: dict[str, list[str]] = {}

        for doc_type, keyword_patterns in self._compiled.items():
            hits = []
            for original_kw, pattern in keyword_patterns:
                if pattern.search(text_lower):
                    hits.append(original_kw)

            if hits:
                total_keywords = len(DOCUMENT_RULES[doc_type])
                raw_score = len(hits) / total_keywords
                bonus = min(0.20, (len(hits) - 1) * 0.05)

                if doc_type in {"mother_deed", "khata_certificate"}:
                    bonus += 0.05

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

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_items) > 1:
            gap = sorted_items[0][1] - sorted_items[1][1]
            if gap < 0.10:
                best_score = max(0.0, best_score - 0.15)

        return {
            "doc_type": best_type,
            "confidence": round(best_score, 4),
            "matched_keywords": matched_map.get(best_type, []),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
            "method": "rule_engine",
        }