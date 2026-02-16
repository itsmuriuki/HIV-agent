"""
Search tools for the agent.
Adapted from the Kenya ARV Guidelines notebook (cells 46).
"""

from typing import List


# Global variables to store the vectorstore and table
# These will be set by search_agent.py during initialization
faq_index = None
faq_vindex = None
embeddings = None


def set_search_index(vectorstore, table, embedding_model):
    """
    Set the global search index variables.
    Called by search_agent.py during initialization.
    
    Args:
        vectorstore: LanceDB vectorstore instance
        table: LanceDB table instance
        embedding_model: HuggingFace embeddings model
    """
    global faq_index, faq_vindex, embeddings
    faq_index = vectorstore
    faq_vindex = table
    embeddings = embedding_model


def text_search(query: str) -> List[str]:
    """
    Perform a text-based search on the document index.
    From notebook cell 46.
    
    This is the tool function that will be used by the pydantic-ai agent.
    
    Args:
        query (str): Search query related to document content
    
    Returns:
        List[str]: A list of up to 5 search results from the index as plain text.
    """
    if faq_index is None:
        return ["Error: Search index not initialized. Please ensure the agent was initialized with a valid index."]
    
    try:
        # Specify search_type="similarity" to use vector similarity search
        # From notebook cell 46
        docs = faq_index.search(query, k=5, search_type="similarity")
        
        # Convert Document objects to plain text
        return [doc.page_content for doc in docs]
    except Exception as e:
        return [f"Error during search: {str(e)}"]


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
    if embeddings is None or faq_vindex is None:
        return []
    
    # 1. Embed query
    query_vector = embeddings.embed_query(query)
    
    # 2. Ensure it is a plain Python list (important!)
    if not isinstance(query_vector, list):
        query_vector = query_vector.tolist()
    
    # 3. Search LanceDB table
    results = (
        faq_vindex
        .search(query_vector)
        .limit(k)
        .to_list()
    )
    
    return results