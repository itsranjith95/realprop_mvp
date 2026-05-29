"""
Ollama HTTP client for LLM-assisted document classification.
Supports: mistral, llama3, gemma2, phi3 — any model you pull locally.
"""

import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are an expert in Indian property documents and legal records.

Given the following extracted text from a scanned document, classify it into exactly ONE of these categories:

Categories:
- sale_deed
- mother_deed
- khata_certificate
- khata_extract
- encumbrance_certificate
- property_tax_receipt
- aadhar_card
- pan_card
- power_of_attorney
- release_deed
- gift_deed
- will
- building_plan
- occupancy_certificate
- unknown

Rules:
1. Reply ONLY with a valid JSON object — no explanation, no markdown, no extra text.
2. JSON format: {{"doc_type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one line>"}}
3. Use "unknown" if you cannot determine the document type with reasonable confidence.
4. Confidence must be a float between 0.0 and 1.0.

Document text (first 2000 characters):
---
{text}
---

JSON response:"""


class OllamaClient:
    """
    Wraps Ollama's /api/generate endpoint.
    Default model: mistral (lightweight, fast, good at structured output).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Return list of pulled models."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def classify(self, text: str) -> dict:
        """
        Send text to Ollama and parse classification response.
        Returns:
            {
                "doc_type": str,
                "confidence": float,
                "reasoning": str,
                "method": "ollama",
                "model": str,
                "error": Optional[str]
            }
        """
        prompt = CLASSIFICATION_PROMPT.format(text=text[:2000])

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temp for deterministic classification
                "top_p": 0.9,
                "num_predict": 120,
            },
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
            return self._parse_response(raw_text)

        except httpx.TimeoutException:
            logger.error("Ollama request timed out after %ds", self.timeout)
            return self._error_result("timeout")
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error: %s", e)
            return self._error_result(f"http_error_{e.response.status_code}")
        except Exception as e:
            logger.error("Ollama unexpected error: %s", e)
            return self._error_result(str(e))

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from Ollama response, tolerating markdown fences."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        # Find first JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("No JSON found in Ollama response: %s", raw[:200])
            return self._error_result("no_json_in_response")

        try:
            data = json.loads(cleaned[start:end])
            return {
                "doc_type": data.get("doc_type", "unknown"),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", ""),
                "method": "ollama",
                "model": self.model,
                "error": None,
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse Ollama JSON: %s | raw: %s", e, raw[:200])
            return self._error_result("json_parse_error")

    def _error_result(self, error: str) -> dict:
        return {
            "doc_type": "unknown",
            "confidence": 0.0,
            "reasoning": "",
            "method": "ollama",
            "model": self.model,
            "error": error,
        }