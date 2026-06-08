"""
RAG Service — Knowledge Base Embedding + Retrieval
Uses HuggingFace Inference API (free tier) for embeddings so no large local model needed.
Falls back to a simple TF-IDF keyword search if API is unavailable.
Stores vectors in a FAISS flat index + SQLite metadata.
"""
import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

import requests
import yaml

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
KB_DIR = Path("kb")
INDEX_DIR = Path("data/kb_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = INDEX_DIR / "kb_metadata.db"
FAISS_PATH = INDEX_DIR / "kb_vectors.faiss"
CHUNKS_PATH = INDEX_DIR / "kb_chunks.json"

# ─── HuggingFace Inference API config ─────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY", "")
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dim

# ─── SQLite metadata helper ───────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_chunks (
            chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            heading    TEXT,
            chunk_text TEXT NOT NULL,
            char_start INTEGER,
            char_end   INTEGER
        )
    """)
    conn.commit()
    return conn


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_markdown(md_text: str, source_file: str, chunk_size: int = 400) -> List[Dict]:
    """Split markdown into heading-level chunks, max ~chunk_size chars."""
    chunks = []
    current_heading = ""
    current_text = ""
    char_start = 0

    for line in md_text.split("\n"):
        if line.startswith("#"):
            if current_text.strip():
                chunks.append({
                    "source_file": source_file,
                    "heading": current_heading,
                    "chunk_text": current_text.strip(),
                    "char_start": char_start,
                    "char_end": char_start + len(current_text),
                })
                char_start += len(current_text)
                current_text = ""
            current_heading = line.strip("# ").strip()
        else:
            current_text += line + "\n"
            if len(current_text) > chunk_size:
                chunks.append({
                    "source_file": source_file,
                    "heading": current_heading,
                    "chunk_text": current_text.strip(),
                    "char_start": char_start,
                    "char_end": char_start + len(current_text),
                })
                char_start += len(current_text)
                current_text = ""

    if current_text.strip():
        chunks.append({
            "source_file": source_file,
            "heading": current_heading,
            "chunk_text": current_text.strip(),
            "char_start": char_start,
            "char_end": char_start + len(current_text),
        })
    return chunks


# ─── Embedding helpers ────────────────────────────────────────────────────────

def _embed_via_hf_api(texts: List[str]) -> Optional[List[List[float]]]:
    """Call HuggingFace Inference API for sentence embeddings."""
    if not HF_API_TOKEN:
        logger.warning("HUGGINGFACE_API_KEY not set — falling back to TF-IDF search.")
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"HF API embedding failed: {e}")
        return None


def _tfidf_search(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """Simple keyword-based fallback search when embeddings unavailable."""
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        text_words = set(chunk["chunk_text"].lower().split())
        score = len(query_words & text_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k] if _ > 0]


# ─── Index build ──────────────────────────────────────────────────────────────

def build_kb_index() -> int:
    """
    Read all .md files in kb/, chunk them, embed via HF API (or skip),
    and store in FAISS + SQLite. Returns number of chunks indexed.
    """
    all_chunks: List[Dict] = []
    for md_file in sorted(KB_DIR.glob("*.md")):
        if md_file.name == "README.md":
            continue
        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text, md_file.name)
        all_chunks.extend(chunks)
        logger.info(f"Chunked {md_file.name} → {len(chunks)} chunks")

    if not all_chunks:
        logger.warning("No KB chunks found. Check kb/ directory.")
        return 0

    # Persist chunks to JSON (always — used for TF-IDF fallback)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    # Save to SQLite
    conn = _get_db()
    conn.execute("DELETE FROM kb_chunks")
    conn.executemany(
        "INSERT INTO kb_chunks (source_file, heading, chunk_text, char_start, char_end) VALUES (?,?,?,?,?)",
        [(c["source_file"], c["heading"], c["chunk_text"], c["char_start"], c["char_end"]) for c in all_chunks],
    )
    conn.commit()
    conn.close()

    # Try to build FAISS index
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = _embed_via_hf_api(texts)
    if embeddings:
        try:
            import faiss
            import numpy as np
            vecs = np.array(embeddings, dtype="float32")
            if vecs.shape[1] != EMBED_DIM:
                logger.warning(f"Unexpected embedding dim {vecs.shape[1]}, skipping FAISS.")
            else:
                index = faiss.IndexFlatIP(EMBED_DIM)
                faiss.normalize_L2(vecs)
                index.add(vecs)
                faiss.write_index(index, str(FAISS_PATH))
                logger.info(f"FAISS index built with {index.ntotal} vectors.")
        except ImportError:
            logger.warning("faiss-cpu not installed — FAISS index skipped. TF-IDF fallback active.")
    else:
        logger.info("Embeddings unavailable — TF-IDF fallback will be used for retrieval.")

    logger.info(f"KB index built: {len(all_chunks)} chunks total.")
    return len(all_chunks)


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieve top_k relevant KB chunks for a given query.
    Uses FAISS (semantic) if available, else TF-IDF keyword fallback.
    Returns list of dicts with: source_file, heading, chunk_text.
    """
    if not CHUNKS_PATH.exists():
        logger.warning("KB index not built. Run build_kb_index() first.")
        return []

    with open(CHUNKS_PATH) as f:
        all_chunks = json.load(f)

    # Try FAISS semantic search
    if FAISS_PATH.exists() and HF_API_TOKEN:
        try:
            import faiss
            import numpy as np
            query_emb = _embed_via_hf_api([query])
            if query_emb:
                index = faiss.read_index(str(FAISS_PATH))
                q_vec = np.array(query_emb, dtype="float32")
                faiss.normalize_L2(q_vec)
                scores, ids = index.search(q_vec, top_k)
                results = []
                for i, score in zip(ids[0], scores[0]):
                    if i < len(all_chunks):
                        chunk = dict(all_chunks[i])
                        chunk["similarity_score"] = float(score)
                        results.append(chunk)
                return results
        except Exception as e:
            logger.warning(f"FAISS search failed: {e} — falling back to TF-IDF.")

    # TF-IDF fallback
    return _tfidf_search(query, all_chunks, top_k=top_k)


def format_context_for_prompt(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context block for the RAG prompt."""
    if not chunks:
        return "No relevant knowledge base passages found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        heading = chunk.get("heading", "")
        source = chunk.get("source_file", "")
        text = chunk.get("chunk_text", "")
        parts.append(f"[{i}] Source: {source} | Section: {heading}\n{text}")
    return "\n\n---\n\n".join(parts)


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild KB index from kb/ folder")
    parser.add_argument("--query", type=str, help="Test retrieval with a query")
    args = parser.parse_args()

    if args.rebuild:
        n = build_kb_index()
        print(f"Index built: {n} chunks.")

    if args.query:
        results = retrieve(args.query, top_k=3)
        print(f"\nTop {len(results)} results for: '{args.query}'\n")
        for r in results:
            print(f"  [{r.get('source_file')}] {r.get('heading')}")
            print(f"  {r.get('chunk_text')[:200]}...\n")