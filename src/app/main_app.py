import streamlit as st

st.set_page_config(page_title="RealProp MVP", layout="wide")
st.title("RealProp MVP")
st.text_input("Case ID")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.button("Mother Deed")
col2.button("Khata")
col3.button("Upload")
col4.button("Cancel")
col5.button("Process")
col6.button("Download Report", disabled=True)

st.info("Phase 0 scaffold is ready.")
