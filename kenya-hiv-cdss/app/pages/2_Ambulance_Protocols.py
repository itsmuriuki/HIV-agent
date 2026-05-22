import os

import streamlit as st

import ui_common


ui_common.load_env()
MODEL = os.getenv("MODEL", "openai:gpt-4o-mini")

st.set_page_config(page_title="Ambulance Protocols", page_icon="🚑", layout="centered")
ui_common.sidebar(MODEL)

st.title("🚑 Ambulance Protocols Assistant")
st.caption(
    "Ask questions about ambulance emergency dispatch and prehospital care protocols (triage, trauma, cardiac arrest, breathing problems, stroke, poisoning, and more)."
)

AMBULANCE_PDF_PATHS = [
    str(ui_common.PROJECT_DIR / "Ambulensi Emergency Medical Dispatch Protocols (1).pdf"),
    str(ui_common.PROJECT_DIR / "Ambulensi Prehospital Emergency Care Clinical Protocols (1).pdf"),
]


@st.cache_resource
def get_agent():
    return ui_common.init_pdf_agent_multi(
        model=MODEL,
        pdf_paths=AMBULANCE_PDF_PATHS,
        db_prefix="ambulance",
        repo_owner="Ambulance",
        repo_name="Emergency-Protocols",
        loading_text="🔄 Loading and indexing ambulance protocols PDFs...",
        ready_text="✅ Ready to answer ambulance protocol questions!",
    )

messages_key = "ambulance_messages"
last_key = "ambulance_last_response"

if messages_key not in st.session_state:
    st.session_state[messages_key] = []

if st.button("🗑️ Clear Ambulance Chat"):
    st.session_state[messages_key] = []
    st.rerun()

if len(st.session_state[messages_key]) == 0:
    st.markdown("### 💡 Try asking:")
    col1, col2 = st.columns(2)
    sample_questions = [
        "What questions should dispatch ask for a suspected stroke call?",
        "What is the protocol for adult cardiac arrest (prehospital)?",
        "How should hypoglycemia be managed prehospital?",
        "What are the priorities for trauma scene management?",
    ]
    for i, question in enumerate(sample_questions):
        col = col1 if i % 2 == 0 else col2
        if col.button(question, key=f"ambulance_sample_{i}"):
            st.session_state[messages_key].append({"role": "user", "content": question})
            st.rerun()

for msg in st.session_state[messages_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question about ambulance protocols...", key="ambulance_input"):
    st.session_state[messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response_text = st.write_stream(ui_common.stream_response(get_agent(), prompt, last_key))
    final_text = st.session_state.get(last_key, response_text)
    st.session_state[messages_key].append({"role": "assistant", "content": final_text})

