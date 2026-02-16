import streamlit as st
import asyncio
import ingest
import search_agent
import logs

# --- Configuration ---
PDF_PATH = "Kenya-ARV-Guidelines-2022-Final-1.pdf"  # Update this to your HIV guideline PDF
REPO_OWNER = "MOH-Kenya"  # Update as needed
REPO_NAME = "HIV-Guidelines"  # Update as needed

# --- Initialization ---
@st.cache_resource
def init_agent():
    st.write("🔄 Loading and indexing HIV guidelines PDF...")
    
    # Index the PDF document
    index = ingest.index_data(PDF_PATH)
    
    # Initialize the agent
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
    
    st.write("✅ Ready to answer questions!")
    return agent

agent = init_agent()

# --- Streamlit UI ---
st.set_page_config(
    page_title="HIV Guidelines Assistant", 
    page_icon="🏥", 
    layout="centered"
)

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
    
    if st.button("📊 View Recent Logs"):
        logs.print_recent_logs(5)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Streaming helper ---
def stream_response(prompt: str):
    async def agen():
        async with agent.run_stream(user_prompt=prompt) as result:
            last_len = 0
            full_text = ""
            async for chunk in result.stream_output(debounce_by=0.01):
                # stream only the delta
                new_text = chunk[last_len:]
                last_len = len(chunk)
                full_text = chunk
                if new_text:
                    yield new_text
            # log once complete
            logs.log_interaction_to_file(agent, result.new_messages())
            st.session_state._last_response = full_text
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen_obj = agen()
    try:
        while True:
            piece = loop.run_until_complete(agen_obj.__anext__())
            yield piece
    except StopAsyncIteration:
        return

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
    
    # Save full response to history
    final_text = getattr(st.session_state, "_last_response", response_text)
    st.session_state.messages.append({"role": "assistant", "content": final_text})