# Knowledge Base (KB) — RealProp MVP

This folder contains curated Markdown snippets about Bengaluru/Karnataka property rules
used as the retrieval corpus for the RAG (Retrieval-Augmented Generation) pipeline.

## Files
| File | Contents |
|---|---|
| `mother_deed_rules.md` | Mother Deed requirements, mandatory fields, red flags |
| `khata_rules.md` | Khata types, BBMP requirements, red flags |
| `bengaluru_property_law_snippets.md` | Stamp duty, EC, RERA, KLR Act snippets |

## Embedding & Indexing
- KB is embedded using HuggingFace `sentence-transformers` via OpenRouter API (or local Mistral).
- Vectors stored in `data/kb_index/` as a FAISS flat index + SQLite metadata table.
- Index is rebuilt via `src/services/rag_service.py:build_kb_index()`.

## Adding New Snippets
1. Add a `.md` file to this folder.
2. Run `python -m src.services.rag_service --rebuild` to re-embed and re-index.
3. Bump `rule_set_version` in `config/rules_config.yaml` if rules change.

## Governance
- KB content is versioned with git (plain Markdown).
- Each snippet should include a source reference or applicable law section.