import os

import streamlit as st

import ui_common


ui_common.load_env()
MODEL = os.getenv("MODEL", "openai:gpt-4o-mini")

st.set_page_config(page_title="HIV Guidelines", page_icon="🏥", layout="centered")
ui_common.sidebar(MODEL)

st.title("🏥 HIV Guidelines Assistant")
st.caption("Ask questions about HIV/AIDS treatment and ARV guidelines (Kenya 2022).")

HIV_PDF_PATH = str(ui_common.PROJECT_DIR / "Kenya-ARV-Guidelines-2022-Final-1.pdf")


@st.cache_resource
def _agent():
    return ui_common.init_pdf_agent_single(
        model=MODEL,
        pdf_path=HIV_PDF_PATH,
        db_prefix="hiv",
        repo_owner="MOH-Kenya",
        repo_name="HIV-Guidelines",
        loading_text="🔄 Loading and indexing HIV guidelines PDF...",
        ready_text="✅ Ready to answer HIV guideline questions!",
    )


agent = _agent()

messages_key = "hiv_messages"
last_key = "hiv_last_response"

if messages_key not in st.session_state:
    st.session_state[messages_key] = []

if st.button("🗑️ Clear HIV Chat"):
    st.session_state[messages_key] = []
    st.rerun()

if len(st.session_state[messages_key]) == 0:
    st.markdown("### 💡 Try asking:")
    col1, col2 = st.columns(2)
    sample_questions = [
        "What are the first-line ART regimens for adults?",
        "When should ART be initiated after HIV diagnosis?",
        "What are the eligibility criteria for starting ART?",
        "How should pregnant women with HIV be treated?",
    ]
    for i, question in enumerate(sample_questions):
        col = col1 if i % 2 == 0 else col2
        if col.button(question, key=f"hiv_sample_{i}"):
            st.session_state[messages_key].append({"role": "user", "content": question})
            st.rerun()

for msg in st.session_state[messages_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question about HIV guidelines...", key="hiv_input"):
    st.session_state[messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response_text = st.write_stream(ui_common.stream_response(agent, prompt, last_key))
    final_text = st.session_state.get(last_key, response_text)
    st.session_state[messages_key].append({"role": "assistant", "content": final_text})

