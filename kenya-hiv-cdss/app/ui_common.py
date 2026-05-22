import asyncio
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

import ingest
import logs
import search_agent


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """
    Run an async coroutine on a persistent event loop.

    Using `asyncio.run()` inside Streamlit can close the loop while async clients (e.g. httpx/anyio)
    are still cleaning up, causing `RuntimeError: Event loop is closed`.
    """
    global _EVENT_LOOP
    if _EVENT_LOOP is None or _EVENT_LOOP.is_closed():
        _EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_EVENT_LOOP)
    return _EVENT_LOOP.run_until_complete(coro)


def load_env():
    # Local: .env file. Cloud: Streamlit secrets (dashboard) — not committed to git.
    load_dotenv(PROJECT_DIR / ".env")
    _hydrate_env_from_streamlit_secrets()


def _hydrate_env_from_streamlit_secrets():
    """Copy Streamlit Cloud secrets into os.environ for OpenAI/LangChain clients."""
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "MODEL",
        "EMBEDDINGS_BACKEND",
    ):
        if os.getenv(key):
            continue
        value = _read_streamlit_secret(key)
        if value:
            os.environ[key] = value


def _read_streamlit_secret(name: str) -> str | None:
    try:
        if hasattr(st.secrets, "get"):
            value = st.secrets.get(name)
            if value:
                return str(value)
        return str(st.secrets[name])
    except (StreamlitSecretNotFoundError, KeyError, TypeError):
        return None


def _get_secret_or_env(secret_name: str) -> str | None:
    return _read_streamlit_secret(secret_name) or os.getenv(secret_name)


def ensure_provider_api_key_or_stop(model: str):
    if model.startswith("openai:"):
        if not _get_secret_or_env("OPENAI_API_KEY"):
            st.error(
                "Missing OpenAI API key.\n\n"
                "**On Streamlit Cloud:** open your app at [share.streamlit.io](https://share.streamlit.io) → "
                "**⚙️ Settings → Secrets** and add:\n\n"
                "```toml\n"
                "OPENAI_API_KEY = \"sk-...\"\n"
                "```\n\n"
                "Click **Save**, then **Reboot app**.\n\n"
                "**Locally:** add `OPENAI_API_KEY=...` to `kenya-hiv-cdss/.env` or "
                "`.streamlit/secrets.toml`, then restart."
            )
            st.stop()

    if model.startswith("anthropic:"):
        if not _get_secret_or_env("ANTHROPIC_API_KEY"):
            st.error(
                "Missing Anthropic API key.\n\n"
                "Set it in your `.env` as `ANTHROPIC_API_KEY=...` (or via Streamlit secrets), then restart the app."
            )
            st.stop()

    if model.startswith("deepseek:"):
        if not _get_secret_or_env("DEEPSEEK_API_KEY"):
            st.error(
                "Missing DeepSeek API key.\n\n"
                "Set it via:\n"
                "- `.env`: `DEEPSEEK_API_KEY=...`\n"
                "- Streamlit secrets: `DEEPSEEK_API_KEY = \"...\"`\n\n"
                "Then restart the app."
            )
            st.stop()

    if model.startswith("gemini:"):
        api_key = _get_secret_or_env("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error(
                "Missing Gemini API key.\n\n"
                "Set it via:\n"
                "- `.env`: `GEMINI_API_KEY=...`\n"
                "- Streamlit secrets: `GEMINI_API_KEY = \"...\"`\n\n"
                "Then restart the app."
            )
            st.stop()
        # Some clients expect GOOGLE_API_KEY.
        os.environ.setdefault("GOOGLE_API_KEY", api_key)


def sidebar(model: str):
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
This is a document-grounded clinical assistant.

Use the navigation to switch between:
- **HIV Guidelines**
- **Ambulance Protocols**
"""
        )
        st.divider()
        if st.button("📊 View Recent Logs"):
            logs.print_recent_logs(5)


def stream_response(agent, prompt: str, response_key: str):
    """
    Streamlit-friendly response helper.

    We avoid driving async streaming contexts manually because it can conflict with AnyIO cancel scopes
    in Streamlit's runtime. Instead, we run the agent synchronously and yield the final text as a
    single chunk (still compatible with st.write_stream).
    """
    try:
        result = _run_async(agent.run(prompt))
        logs.log_interaction_to_file(agent, result.new_messages())
        st.session_state[response_key] = result.output
        yield result.output
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            st.error(
                "Your model provider returned a quota/billing error.\n\n"
                "Check billing/limits for the API key, or switch provider/model via `MODEL` in `.env`."
            )
            return
        raise


def init_pdf_agent_single(
    *,
    model: str,
    pdf_path: str,
    db_prefix: str,
    repo_owner: str,
    repo_name: str,
    loading_text: str,
    ready_text: str,
):
    ensure_provider_api_key_or_stop(model)
    st.write(loading_text)
    db_path = str(PROJECT_DIR / "lancedb" / f"{db_prefix}_{ingest.embeddings_backend_name()}")
    index = ingest.index_data(pdf_path, db_path=db_path)
    agent = search_agent.init_agent(index, repo_owner, repo_name, model=model)
    st.write(ready_text)
    return agent


def init_pdf_agent_multi(
    *,
    model: str,
    pdf_paths: list[str],
    db_prefix: str,
    repo_owner: str,
    repo_name: str,
    loading_text: str,
    ready_text: str,
):
    ensure_provider_api_key_or_stop(model)
    st.write(loading_text)
    db_path = str(PROJECT_DIR / "lancedb" / f"{db_prefix}_{ingest.embeddings_backend_name()}")
    index = ingest.index_pdfs(pdf_paths, db_path=db_path)
    agent = search_agent.init_agent(index, repo_owner, repo_name, model=model)
    st.write(ready_text)
    return agent

