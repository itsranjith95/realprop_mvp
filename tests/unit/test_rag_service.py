"""
Unit Tests — RAG Service (KB chunking, indexing, retrieval)
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def tmp_kb(tmp_path):
    """Create a temporary kb/ directory with sample markdown files."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "test_rules.md").write_text(
        "# Khata Rules\n\nKhata owner name must match buyer name in Mother Deed.\n\n"
        "# Mother Deed Rules\n\nRegistration date is mandatory for all deeds.\n",
        encoding="utf-8",
    )
    return kb_dir, tmp_path


class TestKBChunking:

    def test_chunk_markdown_produces_chunks(self, tmp_kb):
        kb_dir, _ = tmp_kb
        import src.services.rag_service as rag
        text = (kb_dir / "test_rules.md").read_text()
        chunks = rag._chunk_markdown(text, "test_rules.md")
        assert len(chunks) >= 2

    def test_chunk_has_required_fields(self, tmp_kb):
        kb_dir, _ = tmp_kb
        import src.services.rag_service as rag
        text = (kb_dir / "test_rules.md").read_text()
        chunks = rag._chunk_markdown(text, "test_rules.md")
        for chunk in chunks:
            assert "source_file" in chunk
            assert "chunk_text" in chunk
            assert "heading" in chunk
            assert chunk["chunk_text"].strip()

    def test_chunk_source_file_is_set(self, tmp_kb):
        kb_dir, _ = tmp_kb
        import src.services.rag_service as rag
        text = (kb_dir / "test_rules.md").read_text()
        chunks = rag._chunk_markdown(text, "test_rules.md")
        for chunk in chunks:
            assert chunk["source_file"] == "test_rules.md"


class TestTFIDFFallback:

    def test_tfidf_returns_relevant_result(self):
        import src.services.rag_service as rag
        chunks = [
            {"source_file": "a.md", "heading": "Khata", "chunk_text": "Khata owner name must match buyer name"},
            {"source_file": "b.md", "heading": "Stamp Duty", "chunk_text": "Stamp duty rates in Karnataka are 5%"},
            {"source_file": "c.md", "heading": "Registration", "chunk_text": "Registration date is mandatory"},
        ]
        results = rag._tfidf_search("Khata owner name", chunks, top_k=2)
        assert len(results) >= 1
        assert "Khata" in results[0]["chunk_text"] or "owner" in results[0]["chunk_text"]

    def test_tfidf_returns_empty_for_unrelated_query(self):
        import src.services.rag_service as rag
        chunks = [
            {"source_file": "a.md", "heading": "Khata", "chunk_text": "Khata owner name"},
        ]
        results = rag._tfidf_search("blockchain cryptocurrency nft", chunks, top_k=3)
        assert results == []


class TestFormatContext:

    def test_format_context_includes_heading_and_source(self):
        import src.services.rag_service as rag
        chunks = [
            {"source_file": "khata_rules.md", "heading": "Khata Rules",
             "chunk_text": "Khata owner must match deed buyer."}
        ]
        ctx = rag.format_context_for_prompt(chunks)
        assert "khata_rules.md" in ctx
        assert "Khata Rules" in ctx
        assert "Khata owner" in ctx

    def test_format_context_empty_returns_message(self):
        import src.services.rag_service as rag
        ctx = rag.format_context_for_prompt([])
        assert "No relevant" in ctx