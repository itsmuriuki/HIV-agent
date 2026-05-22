"""
Search tools for the agent.
Adapted from the Kenya ARV Guidelines notebook (cells 46).
"""

from typing import List, Callable
from pathlib import Path


def _format_doc_with_ref(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    source = md.get("source") or md.get("document_title") or ""
    page = md.get("page")
    if source:
        source = Path(str(source)).name
    ref = ""
    if source and page is not None:
        ref = f"[{source}][{page}]"
    elif source:
        ref = f"[{source}]"
    elif page is not None:
        ref = f"[page {page}]"
    header = f"{ref}\n" if ref else ""
    return f"{header}{doc.page_content}"


def build_text_search(vectorstore, k: int = 5) -> Callable[[str], List[str]]:
    """
    Build a search tool bound to a specific vectorstore.

    This avoids global mutable state so multiple apps/pages can't overwrite each other.
    """

    def text_search(query: str) -> List[str]:
        try:
            docs = vectorstore.search(query, k=k, search_type="similarity")
            return [_format_doc_with_ref(doc) for doc in docs]
        except Exception as e:
            return [f"Error during search: {str(e)}"]

    return text_search


def vector_search(query: str, k: int = 10) -> List[dict]:
    """
    Perform vector similarity search directly on the LanceDB table.
    From notebook cell 25.
    
    Args:
        query: Search query
        k: Number of results to return
        
    Returns:
        List of result dictionaries
    """
    raise NotImplementedError(
        "vector_search is not used by the Streamlit app. Use build_text_search(vectorstore) instead."
    )