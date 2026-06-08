"""
Unit Tests — Prompt Service (template loading, rendering, LLM call stubs)
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def tmp_prompts(tmp_path, monkeypatch):
    """Point PROMPTS_DIR to a temp directory with test templates."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "test_template.txt").write_text(
        "Hello {{name}}, your rule is {{rule_id}}.", encoding="utf-8"
    )
    import src.services.prompt_service as ps
    monkeypatch.setattr(ps, "PROMPTS_DIR", prompts_dir)
    return prompts_dir, ps


class TestPromptLoading:

    def test_load_prompt_with_txt_extension(self, tmp_prompts):
        _, ps = tmp_prompts
        template = ps.load_prompt_template("test_template.txt")
        assert "{{name}}" in template

    def test_load_prompt_without_extension(self, tmp_prompts):
        _, ps = tmp_prompts
        template = ps.load_prompt_template("test_template")
        assert "{{name}}" in template

    def test_load_prompt_missing_raises(self, tmp_prompts):
        _, ps = tmp_prompts
        with pytest.raises(FileNotFoundError):
            ps.load_prompt_template("nonexistent_template")


class TestPromptRendering:

    def test_render_substitutes_variables(self, tmp_prompts):
        _, ps = tmp_prompts
        result = ps.render_prompt("test_template", {"name": "Ranjith", "rule_id": "R010"})
        assert result == "Hello Ranjith, your rule is R010."

    def test_render_marks_missing_variables(self, tmp_prompts):
        _, ps = tmp_prompts
        result = ps.render_prompt("test_template", {"name": "Ranjith"})
        assert "[MISSING:rule_id]" in result

    def test_render_handles_empty_variables(self, tmp_prompts):
        _, ps = tmp_prompts
        result = ps.render_prompt("test_template", {})
        assert "[MISSING:name]" in result
        assert "[MISSING:rule_id]" in result


class TestLLMCallStubs:

    def test_call_llm_openrouter_no_key_returns_stub(self, tmp_prompts):
        _, ps = tmp_prompts
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
            ps.OPENROUTER_API_KEY = ""
            result = ps.call_llm_openrouter("test prompt")
            assert "unavailable" in result.lower() or "not configured" in result.lower()

    def test_call_llm_openrouter_api_success(self, tmp_prompts):
        _, ps = tmp_prompts
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Explanation text here."}}]
        }
        mock_response.raise_for_status = MagicMock()
        with patch("src.services.prompt_service.requests.post", return_value=mock_response):
            ps.OPENROUTER_API_KEY = "fake_key"
            result = ps.call_llm_openrouter("test prompt")
            assert result == "Explanation text here."

    def test_generate_flag_explanation_returns_string(self, tmp_prompts):
        prompts_dir, ps = tmp_prompts
        # Add the real flag_explanation_prompt.txt to tmp
        (prompts_dir / "flag_explanation_prompt.txt").write_text(
            "Rule: {{rule_id}} — {{rule_name}}. Evidence: {{evidence_summary}}.",
            encoding="utf-8",
        )
        with patch.object(ps, "call_llm", return_value="Test explanation"):
            result = ps.generate_flag_explanation(
                rule_id="R010", rule_name="OWNER_NAME_MISMATCH",
                rule_version="1.1.0", severity="high",
                rule_description="Buyer name mismatch",
                evidence_summary="Ranjith vs Rajesh",
                source_document_id="DOC_001", page=2, bbox="[0,0,100,50]",
                extracted_values="buyer=Ranjith, owner=Rajesh",
            )
            assert result == "Test explanation"