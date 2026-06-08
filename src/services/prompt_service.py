"""
Prompt Service — Load and render prompt templates from config/prompts/.
Supports {{variable}} substitution and calls OpenRouter/Ollama LLM.
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("config/prompts")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL_OPENROUTER = "mistralai/mistral-7b-instruct"
DEFAULT_MODEL_OLLAMA = "mistral:latest"


# ─── Template loading & rendering ────────────────────────────────────────────

def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from config/prompts/<template_name>.
    template_name can omit '.txt' extension.
    """
    fname = template_name if template_name.endswith(".txt") else f"{template_name}.txt"
    path = PROMPTS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template_name: str, variables: Dict[str, Any]) -> str:
    """
    Load template and substitute {{key}} placeholders with values dict.
    Missing keys are left as-is with a [MISSING] marker for debugging.
    """
    template = load_prompt_template(template_name)
    def replacer(match: re.Match) -> str:
        key = match.group(1).strip()
        val = variables.get(key)
        if val is None:
            logger.warning(f"Prompt template '{template_name}': missing variable '{key}'")
            return f"[MISSING:{key}]"
        return str(val)
    return re.sub(r"\{\{(.*?)\}\}", replacer, template)


# ─── LLM call helpers ────────────────────────────────────────────────────────

def call_llm_openrouter(prompt: str, model: str = DEFAULT_MODEL_OPENROUTER, max_tokens: int = 512) -> str:
    """Call OpenRouter API with the rendered prompt. Returns generated text."""
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. Returning stub response.")
        return "[LLM unavailable — OPENROUTER_API_KEY not configured]"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/itsranjith95/realprop_mvp",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        return f"[LLM error: {e}]"


def call_llm_ollama(prompt: str, model: str = DEFAULT_MODEL_OLLAMA, max_tokens: int = 512) -> str:
    """Call local Ollama instance (mistral:latest). Returns generated text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return f"[Ollama error: {e}]"


def call_llm(prompt: str, prefer: str = "openrouter", **kwargs) -> str:
    """
    Unified LLM call. prefer='openrouter' or 'ollama'.
    Falls back to the other if the preferred one fails.
    """
    if prefer == "ollama":
        result = call_llm_ollama(prompt, **kwargs)
        if not result.startswith("[Ollama error"):
            return result
        logger.info("Ollama failed, trying OpenRouter fallback...")
        return call_llm_openrouter(prompt, **kwargs)
    else:
        result = call_llm_openrouter(prompt, **kwargs)
        if not result.startswith("[LLM error") and not result.startswith("[LLM unavail"):
            return result
        logger.info("OpenRouter failed, trying Ollama fallback...")
        return call_llm_ollama(prompt, **kwargs)


# ─── Convenience wrappers ─────────────────────────────────────────────────────

def generate_flag_explanation(
    rule_id: str,
    rule_name: str,
    rule_version: str,
    severity: str,
    rule_description: str,
    evidence_summary: str,
    source_document_id: str,
    page: int,
    bbox: str,
    extracted_values: str,
    prefer_llm: str = "openrouter",
) -> str:
    """Generate a plain-English explanation for a validation flag."""
    variables = {
        "rule_id": rule_id,
        "rule_name": rule_name,
        "rule_version": rule_version,
        "severity": severity,
        "rule_description": rule_description,
        "evidence_summary": evidence_summary,
        "source_document_id": source_document_id,
        "page": page,
        "bbox": bbox,
        "extracted_values": extracted_values,
    }
    prompt = render_prompt("flag_explanation_prompt", variables)
    return call_llm(prompt, prefer=prefer_llm, max_tokens=300)


def generate_case_summary(case_data: Dict[str, Any], prefer_llm: str = "openrouter") -> str:
    """Generate a case-level summary paragraph for the lawyer report."""
    triggered_rules = "\n".join(
        f"  - [{r.get('rule_id')}] {r.get('rule_name')} ({r.get('severity')})"
        for r in case_data.get("triggered_rules", [])
    ) or "  - None"
    agent_findings = "\n".join(
        f"  - {f}" for f in case_data.get("agent_findings", [])
    ) or "  - None"

    variables = {
        "case_id": case_data.get("case_id", ""),
        "property_description": case_data.get("property_description", ""),
        "seller_name": case_data.get("seller_name", ""),
        "buyer_name": case_data.get("buyer_name", ""),
        "khata_owner": case_data.get("khata_owner", ""),
        "registration_date": case_data.get("registration_date", ""),
        "risk_score": case_data.get("risk_score", ""),
        "risk_label": case_data.get("risk_label", ""),
        "triggered_rules_list": triggered_rules,
        "agent_findings": agent_findings,
    }
    prompt = render_prompt("case_summary_prompt", variables)
    return call_llm(prompt, prefer=prefer_llm, max_tokens=400)