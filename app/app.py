import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
import queue
import streamlit as st
import threading
import logs

# Ingest and search_agent are imported inside init_agent() so the app starts
# quickly and passes Cloud health checks; heavy deps load on first use.

# Ensure OPENAI_API_KEY is available (from .env locally or Streamlit Cloud secrets)
if not os.environ.get("OPENAI_API_KEY"):
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    except Exception:
        pass

# --- Configuration ---
PDF_PATH = "Kenya-ARV-Guidelines-2022-Final-1.pdf"  # Update this to your HIV guideline PDF
REPO_OWNER = "MOH-Kenya"  # Update as needed
REPO_NAME = "HIV-Guidelines"  # Update as needed

# --- Streamlit UI (set_page_config must be first Streamlit call) ---
st.set_page_config(
    page_title="HIV Guidelines Assistant",
    page_icon="🏥",
    layout="centered",
)


def _check_api_key():
    """Show a clear error if OPENAI_API_KEY is missing (e.g. on Streamlit Cloud)."""
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "**OpenAI API key is not set.**\n\n"
            "On Streamlit Cloud: open **Manage app** → **Settings** → **Secrets** and add:\n\n"
            "```\nOPENAI_API_KEY = \"your-key-here\"\n```\n\n"
            "Locally: add `OPENAI_API_KEY` to your `.env` file."
        )
        st.stop()

@st.cache_resource
def init_agent():
    import ingest
    import search_agent

    st.write("🔄 Loading and indexing HIV guidelines PDF...")
    
    # Index the PDF document
    index = ingest.index_data(PDF_PATH)
    
    # Initialize the agent
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
    
    st.write("✅ Ready to answer questions!")
    return agent

_check_api_key()

# Lazy init: do not run init_agent() at import so deployment health checks pass.
# Agent is created on first use (first message) via get_agent().

def get_agent():
    """Return the cached agent, initializing on first call."""
    return init_agent()

st.title("🏥 HIV Guidelines Assistant")
st.caption("Ask me anything about HIV/AIDS treatment and ARV guidelines")

# Add sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This assistant helps you find information from the Kenya HIV Prevention 
    and Treatment Guidelines.
    
    **Topics covered:**
    - ART regimens
    - Treatment eligibility
    - Dosing guidelines
    - Special populations
    - Monitoring & follow-up
    - TB/HIV co-infection
    - And more...
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sentinel for end of stream (must not be a string chunk)
_STREAM_END = object()

def stream_response(prompt: str):
    """Stream tokens in real time: run the async agent in a single task in a background thread, yield chunks via queue."""
    agent = get_agent()  # Lazy init on first message
    chunk_queue = queue.Queue()
    result_holder = {}  # full_text, new_messages (set by background thread)

    async def run_stream_and_enqueue():
        full_text = ""
        new_messages = []
        async with agent.run_stream(user_prompt=prompt) as result:
            try:
                async for chunk in result.stream_text(delta=True, debounce_by=0.01):
                    if chunk:
                        full_text += chunk
                        chunk_queue.put(chunk)
            except Exception:
                pass
            if not full_text:
                try:
                    output = await result.get_output()
                    full_text = output if isinstance(output, str) else str(output)
                    if full_text:
                        chunk_queue.put(full_text)
                except Exception:
                    full_text = "Sorry, I couldn't generate a response. Please try again."
                    chunk_queue.put(full_text)
            new_messages = result.new_messages()
        result_holder["full_text"] = full_text
        result_holder["new_messages"] = new_messages
        chunk_queue.put(_STREAM_END)

    def run_in_thread():
        asyncio.run(run_stream_and_enqueue())

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    try:
        while True:
            piece = chunk_queue.get()
            if piece is _STREAM_END:
                break
            yield piece
    finally:
        thread.join()
    # Update session and log on main thread after stream is consumed
    st.session_state._last_response = result_holder.get("full_text", "")
    if result_holder.get("new_messages"):
        logs.log_interaction_to_file(agent, result_holder["new_messages"])

# --- Sample questions ---
if len(st.session_state.messages) == 0:
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
        if col.button(question, key=f"sample_{i}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# --- Chat input ---
if prompt := st.chat_input("Ask your question about HIV guidelines..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistant message (streamed)
    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(prompt))
    
    # Save full response to history (st.write_stream returns None; we store in _last_response)
    final_text = getattr(st.session_state, "_last_response", None) or response_text or ""
    st.session_state.messages.append({"role": "assistant", "content": final_text or "No response generated."})