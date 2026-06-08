"""
Evidence Service — Audit & Evidence Linking
For each validation flag / rule hit, stores and fetches:
  source_document_id, page, bbox, rule_id, rule_version
Writes to the 'evidence' table in SQLite and provides helpers for the UI.
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = Path("data/realprop.db")


# ─── Schema init ─────────────────────────────────────────────────────────────

def init_evidence_table(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create evidence table if it does not exist."""
    _conn = conn or sqlite3.connect(DB_PATH)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id              TEXT    NOT NULL,
            source_document_id   TEXT    NOT NULL,
            doc_type             TEXT,
            page                 INTEGER,
            bbox                 TEXT,
            field_name           TEXT,
            extracted_value      TEXT,
            rule_id              TEXT    NOT NULL,
            rule_version         TEXT    NOT NULL,
            rule_name            TEXT,
            severity             TEXT,
            flag_description     TEXT,
            created_at           TEXT    NOT NULL
        )
    """)
    _conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_evidence_case
        ON evidence (case_id, rule_id)
    """)
    _conn.commit()
    if not conn:
        _conn.close()


# ─── Write helpers ────────────────────────────────────────────────────────────

def store_evidence(
    case_id: str,
    source_document_id: str,
    rule_id: str,
    rule_version: str,
    page: int = 0,
    bbox: Optional[List[float]] = None,
    field_name: str = "",
    extracted_value: str = "",
    doc_type: str = "",
    rule_name: str = "",
    severity: str = "",
    flag_description: str = "",
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Persist one evidence record for a triggered rule flag.
    Returns the new evidence_id.
    """
    _conn = conn or sqlite3.connect(DB_PATH)
    init_evidence_table(_conn)
    bbox_str = str(bbox) if bbox else ""
    now = datetime.now(timezone.utc).isoformat()
    cursor = _conn.execute(
        """
        INSERT INTO evidence
          (case_id, source_document_id, doc_type, page, bbox, field_name,
           extracted_value, rule_id, rule_version, rule_name, severity,
           flag_description, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (case_id, source_document_id, doc_type, page, bbox_str,
         field_name, extracted_value, rule_id, rule_version,
         rule_name, severity, flag_description, now),
    )
    _conn.commit()
    eid = cursor.lastrowid
    if not conn:
        _conn.close()
    logger.debug(f"Evidence stored: evidence_id={eid}, case={case_id}, rule={rule_id}")
    return eid


def store_evidence_batch(records: List[Dict[str, Any]]) -> List[int]:
    """Store multiple evidence records in a single connection."""
    conn = sqlite3.connect(DB_PATH)
    init_evidence_table(conn)
    ids = []
    for rec in records:
        eid = store_evidence(conn=conn, **rec)
        ids.append(eid)
    conn.close()
    return ids


# ─── Read helpers ─────────────────────────────────────────────────────────────

def fetch_evidence_for_case(case_id: str) -> List[Dict[str, Any]]:
    """Return all evidence records for a given case_id, ordered by severity."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_evidence_table(conn)
    rows = conn.execute(
        """
        SELECT * FROM evidence
        WHERE case_id = ?
        ORDER BY CASE severity
            WHEN 'high'   THEN 1
            WHEN 'medium' THEN 2
            WHEN 'low'    THEN 3
            ELSE 4
        END, rule_id
        """,
        (case_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_evidence_for_rule(case_id: str, rule_id: str) -> List[Dict[str, Any]]:
    """Return all evidence records for a specific rule in a case."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_evidence_table(conn)
    rows = conn.execute(
        "SELECT * FROM evidence WHERE case_id=? AND rule_id=? ORDER BY page",
        (case_id, rule_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_evidence_by_doc(case_id: str, source_document_id: str) -> List[Dict[str, Any]]:
    """Return all evidence records for a specific document in a case."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_evidence_table(conn)
    rows = conn.execute(
        "SELECT * FROM evidence WHERE case_id=? AND source_document_id=? ORDER BY page, rule_id",
        (case_id, source_document_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def format_evidence_for_ui(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format evidence records for Streamlit display.
    Returns a list of dicts suitable for st.dataframe() or card rendering.
    """
    formatted = []
    for ev in evidence_list:
        formatted.append({
            "Rule": f"{ev.get('rule_id')} — {ev.get('rule_name', '')}",
            "Severity": ev.get("severity", "").upper(),
            "Field": ev.get("field_name", ""),
            "Extracted Value": ev.get("extracted_value", ""),
            "Document": ev.get("source_document_id", ""),
            "Doc Type": ev.get("doc_type", ""),
            "Page": ev.get("page", ""),
            "BBox": ev.get("bbox", ""),
            "Rule Version": ev.get("rule_version", ""),
            "Description": ev.get("flag_description", ""),
        })
    return formatted