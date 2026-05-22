import os
import streamlit as st

import ui_common

ui_common.load_env()
MODEL = os.getenv("MODEL", "openai:gpt-4o-mini")

st.set_page_config(page_title="Clinical Protocols Assistant", page_icon="🏥", layout="centered")
ui_common.sidebar(MODEL)

st.title("🏥 Clinical Protocols Assistant")
st.markdown(
    """
Use the left navigation to open a dedicated workspace:

- **HIV Guidelines**: Kenya HIV Prevention & Treatment Guidelines (2022)
- **Ambulance Protocols**: dispatch + prehospital clinical protocols
"""
)