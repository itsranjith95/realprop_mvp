"""
Phase 7 – Report Service
Thin wrapper so API routes can call report generation
without importing directly from the pipeline.
"""
from __future__ import annotations

from pathlib import Path

from src.pipelines.review_pipeline import (
    build_report_context,
    record_review_action,
    render_report_to_pdf,
    display_label,
    normalize_label,
)

__all__ = [
    "generate_report",
    "save_review",
    "get_report_path",
]


def generate_report(case_id: str) -> Path:
    """Build context from all artefacts and render PDF. Returns the saved path."""
    ctx = build_report_context(case_id)
    return render_report_to_pdf(ctx)


def save_review(
    case_id: str,
    action: str,
    notes: str = "",
    lawyer_name: str = "Lawyer",
    final_label: str = "",
) -> dict:
    """Persist a lawyer review action and return the record."""
    return record_review_action(
        case_id=case_id,
        action=action,
        notes=notes,
        lawyer_name=lawyer_name,
        final_label=final_label,
    )


def get_report_path(case_id: str, version: int | None = None) -> Path | None:
    """Return path to the latest (or specific version) PDF for a case."""
    from src.pipelines.review_pipeline import REPORTS_DIR

    if version is not None:
        p = REPORTS_DIR / f"{case_id}_v{version}.pdf"
        return p if p.exists() else None

    existing = sorted(
        REPORTS_DIR.glob(f"{case_id}_v*.pdf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return existing[0] if existing else None