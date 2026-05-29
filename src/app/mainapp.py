import json
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
STRICT_MVP_LABELS = ["mother_deed", "khata_certificate"]

st.set_page_config(page_title="RealProp MVP", layout="wide")
st.title("RealProp MVP - Phase 3A")
st.caption("OCR + document classification + manual confirmation")

if "last_pipeline_result" not in st.session_state:
    st.session_state["last_pipeline_result"] = None

with st.sidebar:
    st.subheader("Backend")
    st.write(API_BASE)

    if st.button("Check API Health"):
        try:
            res = requests.get(f"{API_BASE}/health", timeout=5)
            st.success(res.json())
        except Exception as e:
            st.error(str(e))

    if st.button("Check Classification Health"):
        try:
            res = requests.get(f"{API_BASE}/api/v1/classify/health", timeout=10)
            st.success(res.json())
        except Exception as e:
            st.error(str(e))

st.header("Upload and Run Pipeline")

with st.form("ocr_classify_form"):
    case_id = st.text_input("Case ID", value="case01")
    document_type = st.selectbox(
        "Document Type Hint",
        ["motherdeed", "khata", "unknown"],
        index=0,
    )
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "png", "jpg", "jpeg", "webp"]
    )
    submit_pipeline = st.form_submit_button("Run OCR + Classification")

if submit_pipeline:
    if not case_id.strip():
        st.warning("Please enter a case ID.")
    elif uploaded_file is None:
        st.warning("Please upload a file.")
    else:
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }
            data = {
                "case_id": case_id.strip(),
                "document_type": document_type,
            }

            with st.spinner("Running OCR and classification..."):
                res = requests.post(
                    f"{API_BASE}/api/v1/pipeline/ocr-classify",
                    files=files,
                    data=data,
                    timeout=600,
                )

            if res.status_code == 200:
                payload = res.json()
                st.session_state["last_pipeline_result"] = payload
                st.success("Pipeline completed successfully.")
            else:
                try:
                    st.error(res.json())
                except Exception:
                    st.error(res.text)
        except Exception as e:
            st.error(str(e))

result = st.session_state.get("last_pipeline_result")

if result:
    st.header("Prediction")

    classification = result.get("classification", {})
    predicted_label = classification.get("doc_type", "unknown")
    confidence = classification.get("confidence", 0.0)
    needs_review = classification.get("needs_human_review", False)
    method = classification.get("method", "")
    matched_keywords = classification.get("matched_keywords", [])
    reasoning = classification.get("reasoning", "")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Type", predicted_label)
    col2.metric("Confidence", f"{confidence:.2f}")
    col3.metric("Method", method)
    col4.metric("Needs Review", "Yes" if needs_review else "No")

    if matched_keywords:
        st.write("**Matched Keywords:**", ", ".join(matched_keywords))

    if reasoning:
        st.write("**Reasoning:**", reasoning)

    if needs_review or predicted_label not in STRICT_MVP_LABELS:
        st.warning("Low confidence or non-MVP label detected. Manual confirmation required.")

        confirmed_label = st.selectbox(
            "Manual Confirmation",
            STRICT_MVP_LABELS,
            index=0 if predicted_label != "khata_certificate" else 1,
        )
        review_notes = st.text_area("Reviewer Notes", placeholder="Enter review notes")

        if st.button("Approve Review Decision"):
            reviewed = dict(result)
            reviewed["review"] = {
                "confirmed_label": confirmed_label,
                "review_notes": review_notes,
                "review_required": True,
            }
            st.session_state["last_pipeline_result"] = reviewed
            st.success(f"Manual review saved with label: {confirmed_label}")

    tab1, tab2, tab3 = st.tabs(["OCR Text", "OCR JSON", "Final JSON"])

    with tab1:
        aggregated_text = result.get("aggregated_text", "")
        st.text_area("Aggregated OCR Text", aggregated_text, height=300)

    with tab2:
        st.json(result.get("ocr", {}))

    with tab3:
        st.json(st.session_state["last_pipeline_result"])

    st.download_button(
        label="Download Pipeline Result JSON",
        data=json.dumps(st.session_state["last_pipeline_result"], indent=2),
        file_name=f"{result.get('document_id', 'pipeline_result')}.json",
        mime="application/json",
    )