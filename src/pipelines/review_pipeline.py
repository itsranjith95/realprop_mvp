"""
Phase 7 – Review Pipeline
Reads from the ACTUAL artifact paths used by Phases 4, 5, 6:
  - Entities       → SQLite app.db (ExtractedEntity table)
  - Risk/Agent     → data/results/risk_scores/{case_id}_risk.json
  - OCR pages      → data/ocr/{case_id}/{doc_id}/page*.json
  - Reviews        → data/reviews/{case_id}.json
  - Reports        → data/reports/{case_id}_v{n}.pdf
  - Report context → data/report_contexts/{case_id}_v{n}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data")
OCR_DIR = DATA_ROOT / "ocr"
RISK_SCORES_DIR = DATA_ROOT / "results" / "risk_scores"
REVIEWS_DIR = DATA_ROOT / "reviews"
REPORTS_DIR = DATA_ROOT / "reports"
REPORT_CONTEXTS_DIR = DATA_ROOT / "report_contexts"
DB_PATH = DATA_ROOT / "app.db"

for _d in (REVIEWS_DIR, REPORTS_DIR, REPORT_CONTEXTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------
_CANONICAL_LABEL_MAP = {
    "motherdeed": "mother_deed",
    "mother_deed": "mother_deed",
    "mother deed": "mother_deed",
    "khata": "khata_certificate",
    "khata_certificate": "khata_certificate",
    "khata certificate": "khata_certificate",
    "sale_deed": "sale_deed",
    "sale deed": "sale_deed",
    "ec": "ec",
    "encumbrance_certificate": "ec",
    "other": "other",
    "unknown": "unknown",
}


def normalize_label(label: str | None) -> str:
    if not label:
        return "unknown"
    key = str(label).strip().lower()
    return _CANONICAL_LABEL_MAP.get(key, key.replace(" ", "_"))


def display_label(label: str | None) -> str:
    canonical = normalize_label(label)
    return canonical.replace("_", " ").title()


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
    return {}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _fetch_entities_from_db(case_id: str) -> dict:
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                doc_type,
                field_name,
                value,
                normalized_value,
                confidence,
                source_doc,
                page,
                bbox
            FROM extracted_entities
            WHERE case_id = ?
            ORDER BY doc_type, field_name
            """,
            (case_id,),
        )
        rows = cur.fetchall()
        conn.close()

        result: dict = {}
        for row in rows:
            dtype = normalize_label(row["doc_type"])
            if dtype not in result:
                result[dtype] = {}

            result[dtype][row["field_name"]] = {
                "value": row["value"],
                "normalized": row["normalized_value"],
                "confidence": row["confidence"],
                "source_doc": row["source_doc"],
                "page": row["page"],
                "bbox": row["bbox"],
            }
        return result

    except Exception as exc:
        logger.warning("Could not read entities from DB: %s", exc)
        return {}


def _fetch_case_status_from_db(case_id: str) -> str:
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT status FROM cases WHERE id = ?", (case_id,))
        row = cur.fetchone()
        conn.close()
        return row["status"] if row else "unknown"
    except Exception as exc:
        logger.warning("Could not read case status from DB: %s", exc)
        return "unknown"


def _fetch_documents_for_case(case_id: str) -> list[dict]:
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                case_id,
                doc_type,
                original_filename,
                stored_filename,
                path,
                status,
                created_at
            FROM documents
            WHERE case_id = ?
            ORDER BY created_at DESC
            """,
            (case_id,),
        )
        rows = cur.fetchall()
        conn.close()

        docs = []
        for r in rows:
            item = dict(r)
            item["doc_type"] = normalize_label(item.get("doc_type"))
            docs.append(item)
        return docs

    except Exception as exc:
        logger.warning("Could not read documents from DB: %s", exc)
        return []


def _summarize_case_doc_types(case_id: str) -> str:
    docs = _fetch_documents_for_case(case_id)
    if not docs:
        return "unknown"
    unique_types = []
    for d in docs:
        dt = normalize_label(d.get("doc_type"))
        if dt not in unique_types:
            unique_types.append(dt)
    return ", ".join(unique_types)


def _fetch_all_cases_from_db() -> list[dict]:
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, status, created_at
            FROM cases
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
        conn.close()

        cases = []
        for row in rows:
            case_id = row["id"]
            cases.append(
                {
                    "case_id": case_id,
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "doc_type": _summarize_case_doc_types(case_id),
                }
            )
        return cases

    except Exception as exc:
        logger.warning("Could not read cases from DB: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Review action
# ---------------------------------------------------------------------------
def record_review_action(
    case_id: str,
    action: str,
    notes: str = "",
    lawyer_name: str = "Lawyer",
    final_label: str = "",
) -> dict:
    record = {
        "case_id": case_id,
        "action": action,
        "notes": notes,
        "lawyer_name": lawyer_name,
        "final_label": normalize_label(final_label),
        "reviewed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    review_path = REVIEWS_DIR / f"{case_id}.json"
    history: list = []
    if review_path.exists():
        try:
            existing = json.loads(review_path.read_text(encoding="utf-8"))
            history = existing if isinstance(existing, list) else [existing]
        except Exception:
            history = []

    history.append(record)
    review_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Recorded review action '%s' for case '%s'", action, case_id)
    return record


# ---------------------------------------------------------------------------
# Report version helpers
# ---------------------------------------------------------------------------
def _next_report_version(case_id: str) -> int:
    existing = sorted(REPORTS_DIR.glob(f"{case_id}_v*.pdf"))
    if not existing:
        existing = sorted(REPORTS_DIR.glob(f"{case_id}_v*.html"))
    versions = []
    for p in existing:
        stem = p.stem
        if "_v" in stem:
            try:
                versions.append(int(stem.rsplit("_v", 1)[1]))
            except Exception:
                pass
    return (max(versions) + 1) if versions else 1


def _persist_report_context(case_id: str, version: int, context: dict) -> Path:
    out = REPORT_CONTEXTS_DIR / f"{case_id}_v{version}.json"
    out.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Build report context
# ---------------------------------------------------------------------------
def build_report_context(case_id: str) -> dict:
    review_path = REVIEWS_DIR / f"{case_id}.json"
    review_history = _load_json(review_path) if review_path.exists() else []
    latest_review = review_history[-1] if isinstance(review_history, list) and review_history else {}

    risk_path = RISK_SCORES_DIR / f"{case_id}_risk.json"
    risk_output = _load_json(risk_path)
    risk_available = bool(risk_output)

    ctx: dict = {
        "case_id": case_id,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "case_status": _fetch_case_status_from_db(case_id),
        "ocr_pages": [],
        "entities": _fetch_entities_from_db(case_id),
        "risk_output": risk_output,
        "risk_available": risk_available,
        "risk_note": "" if risk_available else "Phase 6 output not available for this case.",
        "review_history": review_history if isinstance(review_history, list) else [],
        "latest_review": latest_review,
        "documents": _fetch_documents_for_case(case_id),
        "doc_type_summary": _summarize_case_doc_types(case_id),
        "report_version": _next_report_version(case_id),
    }

    ocr_case_dir = OCR_DIR / case_id
    if ocr_case_dir.is_dir():
        for doc_dir in sorted(ocr_case_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            for page_file in sorted(doc_dir.glob("page*.json")):
                try:
                    page_data = json.loads(page_file.read_text(encoding="utf-8"))
                    page_data["_source_file"] = str(page_file)
                    ctx["ocr_pages"].append(page_data)
                except Exception as exc:
                    logger.warning("Could not read OCR page %s: %s", page_file, exc)

    return ctx


# ---------------------------------------------------------------------------
# Render report
# ---------------------------------------------------------------------------
def _risk_colour(score) -> tuple:
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    if score >= 70:
        return (0.85, 0.18, 0.18)
    if score >= 40:
        return (0.9, 0.55, 0.05)
    return (0.2, 0.65, 0.3)


def render_report_to_pdf(context: dict) -> Path:
    case_id = context.get("case_id", "unknown")
    version = int(context.get("report_version", 1))
    out_path = REPORTS_DIR / f"{case_id}_v{version}.pdf"

    _persist_report_context(case_id, version, context)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        s_h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
        s_h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=4)
        s_body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)
        s_sm = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, textColor=colors.gray)

        elements = [
            Paragraph("RealProp MVP — Due Diligence Report", s_h1),
            Paragraph(f"Case ID: <b>{case_id}</b>", s_body),
            Paragraph(f"Version: v{version}", s_body),
            Paragraph(f"Document Types: {context.get('doc_type_summary', 'unknown')}", s_body),
            Paragraph(f"Generated: {context.get('generated_at', '')}", s_sm),
            HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
            Spacer(1, 0.3 * cm),
        ]

        risk = context.get("risk_output", {}) or {}
        if context.get("risk_available"):
            score = risk.get("risk_score", 0)
            label = risk.get("risk_label", "unknown")
            summary = risk.get("summary", "")

            elements.append(Paragraph("Risk Assessment", s_h2))
            r, g, b = _risk_colour(score)
            rt = Table(
                [["Risk Score", "Risk Label"], [str(score), str(label).upper()]],
                colWidths=[5 * cm, 5 * cm],
            )
            rt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, 1), colors.Color(r, g, b)),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            elements.append(rt)
            if summary:
                elements += [Spacer(1, 0.2 * cm), Paragraph(f"<i>{summary}</i>", s_sm)]
        else:
            elements.append(Paragraph("Risk Assessment", s_h2))
            elements.append(Paragraph("Phase 6 output not available for this case.", s_body))

        entities = context.get("entities", {}) or {}
        if entities:
            elements += [Spacer(1, 0.3 * cm), Paragraph("Extracted Fields", s_h2)]
            ent_rows = [["Doc Type", "Field", "Value", "Confidence"]]
            for doc_type, fields in entities.items():
                if isinstance(fields, dict):
                    for field, meta in fields.items():
                        if isinstance(meta, dict):
                            ent_rows.append([
                                display_label(doc_type),
                                field,
                                str(meta.get("normalized") or meta.get("value", "")),
                                str(meta.get("confidence", "")),
                            ])
            et = Table(ent_rows, colWidths=[3.5 * cm, 5 * cm, 6.5 * cm, 2.0 * cm])
            et.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ]))
            elements.append(et)

        latest = context.get("latest_review", {}) or {}
        if latest:
            elements += [Spacer(1, 0.3 * cm), Paragraph("Lawyer Review", s_h2)]
            rev_t = Table(
                [
                    ["Reviewer", latest.get("lawyer_name", "—")],
                    ["Action", latest.get("action", "—")],
                    ["Final Label", display_label(latest.get("final_label", "—"))],
                    ["Reviewed At", latest.get("reviewed_at", "—")],
                    ["Notes", latest.get("notes", "—")],
                ],
                colWidths=[4 * cm, 13 * cm],
            )
            rev_t.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            elements.append(rev_t)

        elements += [
            Spacer(1, 0.4 * cm),
            HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey),
            Paragraph("Generated by RealProp MVP — Confidential. For lawyer use only.", s_sm),
        ]

        doc.build(elements)
        logger.info("PDF report saved: %s", out_path)
        return out_path

    except ImportError:
        logger.warning("ReportLab not installed — falling back to HTML report.")
        return _render_html_fallback(context, case_id, version)


def _render_html_fallback(context: dict, case_id: str, version: int) -> Path:
    html_path = REPORTS_DIR / f"{case_id}_v{version}.html"
    risk = context.get("risk_output", {}) or {}
    latest = context.get("latest_review", {}) or {}

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>RealProp Report – {case_id} v{version}</title></head><body>
<h1>RealProp MVP — Report</h1>
<p><b>Case:</b> {case_id} | <b>Version:</b> v{version} | <b>Generated:</b> {context.get('generated_at','')}</p>
<p><b>Document Types:</b> {context.get('doc_type_summary','unknown')}</p>
<h2>Risk</h2>
<p>{'Phase 6 output not available for this case.' if not context.get('risk_available') else f"Score: {risk.get('risk_score','N/A')} | Label: {risk.get('risk_label','unknown')}"}</p>
<p>Summary: {risk.get('summary','')}</p>
<h2>Lawyer Review</h2>
<p>Action: {latest.get('action','—')} | By: {latest.get('lawyer_name','—')}</p>
<p>Final Label: {display_label(latest.get('final_label','—'))}</p>
<p>Notes: {latest.get('notes','—')}</p>
<pre>{json.dumps(context, indent=2, default=str)}</pre>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved: %s", html_path)
    return html_path