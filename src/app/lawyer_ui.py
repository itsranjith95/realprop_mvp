"""
Phase 7 – Lawyer Review UI
Reads from ACTUAL project artifact paths:
  - Cases / entities / documents → SQLite data/app.db
  - Risk output                  → data/results/risk_scores/{case_id}_risk.json
  - OCR pages                    → data/ocr/{case_id}/{doc_id}/page*.json
  - Reviews                      → data/reviews/{case_id}.json
  - Reports                      → data/reports/{case_id}_v{n}.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.review_pipeline import (
    _fetch_all_cases_from_db,
    _fetch_documents_for_case,
    _fetch_entities_from_db,
    build_report_context,
    record_review_action,
    render_report_to_pdf,
    normalize_label,
    display_label,
    RISK_SCORES_DIR,
    REVIEWS_DIR,
    OCR_DIR,
)

ACTION_LABELS = {
    "approve": "✅ Approve",
    "request_clarification": "🔄 Request Clarification",
    "mark_high_risk": "🚨 Mark High Risk",
}

STATUS_COLOUR = {
    "approve": "green",
    "request_clarification": "orange",
    "mark_high_risk": "red",
    "pending": "gray",
    "REVIEW_COMPLETE": "green",
    "PROCESSING": "blue",
    "UPLOADED": "gray",
    "DRAFT": "gray",
}


def _load_json(path: Path) -> dict | list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Case List View
# ---------------------------------------------------------------------------

def render_case_list():
    st.subheader("📋 Case List")

    db_cases = _fetch_all_cases_from_db()
    if not db_cases:
        st.info(
            "No cases found in `data/app.db`. "
            "Create a case via the main app and run it through the pipeline first."
        )
        return

    import pandas as pd

    rows = []
    for c in db_cases:
        case_id = c["case_id"]

        review_history = _load_json(REVIEWS_DIR / f"{case_id}.json")
        if isinstance(review_history, list) and review_history:
            review_action = review_history[-1].get("action", "pending")
        else:
            review_action = "pending"

        risk_data = _load_json(RISK_SCORES_DIR / f"{case_id}_risk.json")
        risk_score = risk_data.get("risk_score", "—")
        risk_label = risk_data.get("risk_label", "—")

        doc_type_summary = c.get("doc_type", "unknown")
        pretty_doc_types = ", ".join(
            display_label(x.strip()) for x in str(doc_type_summary).split(",")
        )

        rows.append({
            "Case ID": case_id,
            "Doc Type": pretty_doc_types,
            "Pipeline Status": c.get("status", "—"),
            "Review Action": review_action,
            "Risk Score": risk_score,
            "Risk Label": risk_label,
            "Created": str(c.get("created_at", ""))[:19],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    case_ids = [c["case_id"] for c in db_cases]
    selected = st.selectbox("Open case detail →", options=["— select —"] + case_ids)
    if selected and selected != "— select —":
        st.session_state["active_case"] = selected
        st.session_state["view"] = "detail"
        st.rerun()


# ---------------------------------------------------------------------------
# Case Detail View
# ---------------------------------------------------------------------------

def render_case_detail(case_id: str):
    st.subheader(f"🗂 Case Detail — `{case_id}`")

    if st.button("← Back to Case List"):
        st.session_state["view"] = "list"
        st.rerun()

    # Load artefacts
    entities = _fetch_entities_from_db(case_id)
    documents = _fetch_documents_for_case(case_id)
    risk_data = _load_json(RISK_SCORES_DIR / f"{case_id}_risk.json")

    review_history = _load_json(REVIEWS_DIR / f"{case_id}.json")
    latest_review = (
        review_history[-1]
        if isinstance(review_history, list) and review_history
        else {}
    )

    # ---- Document Viewer ----
    with st.expander("📄 Document Pages (OCR)", expanded=False):
        ocr_case_dir = OCR_DIR / case_id
        ocr_pages_found = False

        if ocr_case_dir.is_dir():
            for doc_dir in sorted(ocr_case_dir.iterdir()):
                if not doc_dir.is_dir():
                    continue
                page_files = sorted(doc_dir.glob("page*.json"))
                if page_files:
                    st.markdown(f"**Document folder:** `{doc_dir.name}`")
                for pf in page_files:
                    ocr_pages_found = True
                    try:
                        page_data = json.loads(pf.read_text(encoding="utf-8"))
                    except Exception:
                        continue

                    page_idx = page_data.get("page_index", pf.stem)
                    img_path = page_data.get("image_path", "")

                    st.markdown(f"**Page {page_idx}** — `{pf}`")
                    if img_path and Path(img_path).exists():
                        st.image(img_path, use_column_width=True)
                    elif img_path:
                        st.caption(f"Image path: `{img_path}` _(file not found locally)_")

                    text = page_data.get("text", "")
                    if text:
                        with st.expander(f"OCR Text — Page {page_idx}", expanded=False):
                            st.text(text[:2000] + ("…" if len(text) > 2000 else ""))

        if not ocr_pages_found:
            st.info(f"No OCR page JSON files found under `data/ocr/{case_id}/`")

        if documents:
            st.markdown("**Documents in database:**")
            for doc in documents:
                st.markdown(
                    f"- `{doc.get('id','')[:8]}…` | type: **{display_label(doc.get('doc_type','?'))}** "
                    f"| status: {doc.get('status','?')} | path: `{doc.get('path','')}`"
                )

    # ---- Three analysis panels ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔍 Extracted Fields")
        if entities:
            for doc_type, fields in entities.items():
                st.markdown(f"**{display_label(doc_type)}**")
                if isinstance(fields, dict):
                    for field, meta in fields.items():
                        if isinstance(meta, dict):
                            val = meta.get("normalized") or meta.get("value", "—")
                            conf = meta.get("confidence", "")
                            conf_str = f" _(conf: {conf})_" if conf else ""
                            st.markdown(f"- **{field}**: {val}{conf_str}")
                        else:
                            st.markdown(f"- **{field}**: {meta}")
        else:
            st.info("No extracted fields in DB for this case.")

    with col2:
        st.markdown("### ✅ Validation & Rules")
        if risk_data:
            rule_hits = risk_data.get("rule_hits", [])
            overall = risk_data.get("risk_label", "")
            mandatory = risk_data.get("mandatory_review", False)

            if overall:
                colour = (
                    "red" if "high" in str(overall).lower()
                    else "orange" if "medium" in str(overall).lower()
                    else "green"
                )
                st.markdown(f"**Overall Risk:** :{colour}[{overall.upper()}]")

            if mandatory:
                st.warning("⚠️ Mandatory human review required.")

            if rule_hits:
                for hit in rule_hits:
                    sev = hit.get("severity", "")
                    icon = "🔴" if sev == "critical" else "🟡" if sev == "high" else "🟢"
                    st.markdown(
                        f"{icon} **{hit.get('name', hit.get('rule_id', ''))}** "
                        f"— sev: {sev} | pts: {hit.get('points', 0)}"
                    )
            else:
                st.success("No rule violations triggered.")
        else:
            st.info("No risk/validation output found. Run Full Process first.")

    with col3:
        st.markdown("### 🤖 Risk Score")
        if risk_data:
            risk_score = risk_data.get("risk_score", None)
            risk_label = risk_data.get("risk_label", "—")
            summary = risk_data.get("summary", "")

            if risk_score is not None:
                try:
                    score_f = float(risk_score)
                    colour = (
                        "red" if score_f >= 70
                        else "orange" if score_f >= 40
                        else "green"
                    )
                except Exception:
                    colour = "gray"
                st.metric("Risk Score", f"{risk_score}")
                st.markdown(f"Risk Label: :{colour}[**{risk_label.upper()}**]")

            if summary:
                st.markdown(f"**Summary:** {summary}")

            extras = {
                k: v for k, v in risk_data.items()
                if k not in {"risk_score", "risk_label", "summary", "rule_hits",
                              "mandatory_review", "case_id", "doc_type"}
            }
            if extras:
                with st.expander("Full risk JSON", expanded=False):
                    st.json(extras)
        else:
            st.info("No agent/risk output available.")

    st.divider()

    # ---- Current review status banner ----
    if latest_review:
        action = latest_review.get("action", "pending")
        colour = STATUS_COLOUR.get(action, "gray")
        st.markdown(
            f"**Current Status:** :{colour}[{action}] "
            f"&nbsp;|&nbsp; Last reviewed by **{latest_review.get('lawyer_name', '—')}** "
            f"at {latest_review.get('reviewed_at', '—')}"
        )
        if latest_review.get("notes"):
            st.caption(f"Note: {latest_review['notes']}")

    # ---- Lawyer Controls ----
    st.markdown("### 📝 Lawyer Review Controls")

    with st.form(key=f"review_form_{case_id}"):
        lawyer_name = st.text_input("Reviewer Name", value="Advocate")
        final_label = st.selectbox(
            "Confirmed Document Label",
            ["mother_deed", "khata_certificate", "sale_deed", "ec", "other", "unknown"],
            format_func=display_label,
        )
        action_choice = st.radio(
            "Action",
            options=list(ACTION_LABELS.keys()),
            format_func=lambda k: ACTION_LABELS[k],
            horizontal=True,
        )
        comment = st.text_area(
            "Comments / Notes",
            placeholder="Enter observations, concerns, or clarification requests…",
            height=120,
        )
        submitted = st.form_submit_button("💾 Save Review Action")

    if submitted:
        record_review_action(
            case_id=case_id,
            action=action_choice,
            notes=comment,
            lawyer_name=lawyer_name,
            final_label=final_label,
        )
        st.success(f"Review action **{action_choice}** saved successfully.")
        st.rerun()

    # ---- Review History ----
    if isinstance(review_history, list) and len(review_history) > 1:
        with st.expander(f"📜 Review History ({len(review_history)} entries)", expanded=False):
            import pandas as pd
            hist_rows = [
                {
                    "#": i + 1,
                    "Reviewer": r.get("lawyer_name", "—"),
                    "Action": r.get("action", "—"),
                    "Label": display_label(r.get("final_label", "—")),
                    "Notes": r.get("notes", ""),
                    "Timestamp": r.get("reviewed_at", "—"),
                }
                for i, r in enumerate(review_history)
            ]
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ---- Report Generation ----
    st.markdown("### 📑 Generate PDF Report")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        if st.button("🖨 Generate & Download Report", key=f"gen_{case_id}"):
            with st.spinner("Building report…"):
                try:
                    ctx = build_report_context(case_id)
                    report_path = render_report_to_pdf(ctx)
                    st.session_state[f"report_path_{case_id}"] = str(report_path)
                    st.success(f"Report saved: `{report_path}`")
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")

    with col_b:
        report_path_str = st.session_state.get(f"report_path_{case_id}")
        if not report_path_str:
            # Auto-detect latest existing report
            existing = sorted(
                (Path("data/reports")).glob(f"{case_id}_v*.pdf"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if existing:
                report_path_str = str(existing[0])

        if report_path_str:
            rp = Path(report_path_str)
            if rp.exists():
                with open(rp, "rb") as fh:
                    file_bytes = fh.read()
                mime = "application/pdf" if rp.suffix == ".pdf" else "text/html"
                st.download_button(
                    label=f"⬇ Download {rp.name}",
                    data=file_bytes,
                    file_name=rp.name,
                    mime=mime,
                    key=f"dl_{case_id}_{rp.name}",
                )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RealProp – Lawyer Review",
        layout="wide",
        page_icon="⚖️",
    )
    st.title("⚖️ RealProp MVP — Lawyer Review")
    st.caption("Phase 7 · Review cases, add comments, generate PDF reports")

    if "view" not in st.session_state:
        st.session_state["view"] = "list"
    if "active_case" not in st.session_state:
        st.session_state["active_case"] = None

    with st.sidebar:
        st.header("Navigation")
        if st.button("📋 Case List"):
            st.session_state["view"] = "list"
            st.rerun()
        if st.session_state.get("active_case"):
            if st.button(f"🗂 {st.session_state['active_case']}"):
                st.session_state["view"] = "detail"
                st.rerun()
        st.divider()
        st.caption("Artifact paths:")
        st.code(
            "data/app.db\n"
            "data/results/risk_scores/\n"
            "data/ocr/\n"
            "data/reviews/\n"
            "data/reports/"
        )

    if st.session_state["view"] == "detail" and st.session_state.get("active_case"):
        render_case_detail(st.session_state["active_case"])
    else:
        render_case_list()


if __name__ == "__main__":
    main()