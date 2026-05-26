import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RealProp MVP", layout="wide")
st.title("RealProp MVP - Phase 1")
st.caption("Case intake and document ingestion pipeline")

with st.sidebar:
    st.subheader("API")
    st.write(API_BASE)
    if st.button("Check API Health"):
        try:
            res = requests.get(f"{API_BASE}/health", timeout=5)
            st.success(res.json())
        except Exception as e:
            st.error(str(e))

st.header("Create Case")

with st.form("create_case_form"):
    case_id = st.text_input("Case ID (optional)")
    property_type = st.selectbox("Property Type", ["residential", "commercial", "site", "apartment"])
    city = st.text_input("City", value="Bengaluru")
    created_by = st.text_input("Created By", value="internal_user")
    property_description = st.text_area("Property Description")
    create_case_btn = st.form_submit_button("Create Case")

if create_case_btn:
    payload = {
        "case_id": case_id or None,
        "property_type": property_type,
        "city": city,
        "created_by": created_by,
        "property_description": property_description or None,
    }
    try:
        res = requests.post(f"{API_BASE}/api/v1/cases", json=payload, timeout=15)
        if res.status_code == 200:
            case = res.json()["case"]
            st.session_state["active_case_id"] = case["id"]
            st.success(f"Case created: {case['id']}")
        else:
            st.error(res.json().get("detail", "Failed to create case"))
    except Exception as e:
        st.error(str(e))

st.header("Upload Documents")

default_case_id = st.session_state.get("active_case_id", "")
upload_case_id = st.text_input("Case ID for upload", value=default_case_id)
doc_type = st.selectbox("Document Type", ["motherdeed", "khata"])
uploaded_file = st.file_uploader("Choose file", type=["pdf", "png", "jpg", "jpeg"])

if st.button("Upload Document"):
    if not upload_case_id:
        st.warning("Please provide a case ID.")
    elif not uploaded_file:
        st.warning("Please choose a file.")
    else:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")
        }
        data = {
            "doc_type": doc_type,
            "actor_id": "streamlit_user",
        }
        try:
            res = requests.post(
                f"{API_BASE}/api/v1/cases/{upload_case_id}/documents",
                files=files,
                data=data,
                timeout=60,
            )
            if res.status_code == 200:
                doc = res.json()["document"]
                st.success(f"Uploaded: {doc['original_filename']} -> {doc['path']}")
            else:
                st.error(res.json().get("detail", "Upload failed"))
        except Exception as e:
            st.error(str(e))

st.header("Case Lookup")

lookup_case_id = st.text_input("Lookup Case ID", value=default_case_id, key="lookup_case")
if st.button("Fetch Case Details"):
    if not lookup_case_id:
        st.warning("Enter a case ID.")
    else:
        try:
            res = requests.get(f"{API_BASE}/api/v1/cases/{lookup_case_id}", timeout=15)
            if res.status_code == 200:
                payload = res.json()
                st.subheader("Case")
                st.json(payload["case"])
                st.subheader("Documents")
                st.json(payload["documents"])
            else:
                st.error(res.json().get("detail", "Case not found"))
        except Exception as e:
            st.error(str(e))

st.header("All Cases")

if st.button("Refresh Case List"):
    try:
        res = requests.get(f"{API_BASE}/api/v1/cases", timeout=15)
        if res.status_code == 200:
            cases = res.json().get("cases", [])
            st.json(cases)
        else:
            st.error("Could not load cases")
    except Exception as e:
        st.error(str(e))