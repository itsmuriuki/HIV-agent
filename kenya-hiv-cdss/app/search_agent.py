"""
Search agent module using pydantic-ai.
Adapted from the Kenya ARV Guidelines notebook (cells 43, 47, 67).
"""

from pydantic_ai import Agent
import search_tools
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_system_prompt(repo_owner: str, repo_name: str) -> str:
    """
    Create the system prompt for the agent.
    Adapted from notebook cells 43 and 67.
    
    Args:
        repo_owner: Repository owner (for context)
        repo_name: Repository name (for context)
        
    Returns:
        System prompt string
    """
    return f"""
You are a helpful assistant for the {repo_owner}/{repo_name} documents.

Use the search tool to find relevant information from the documents before answering questions.

If the initial search does not return sufficient or precise information:
- Refine the query using relevant keywords and synonyms.
- Perform multiple searches if necessary.
- Combine relevant retrieved sections before forming a response.

When relevant content is found:
- Base your answer strictly and only on the retrieved information.
- Do NOT introduce external knowledge not present in the documents.
- Provide clear, structured answers.
- Always include references by citing the filename and page of the source material you used.

When citing the reference, format it as:
[DOCUMENT TITLE][PAGE]

If no relevant information is found after multiple searches:
- Clearly state: "This information was not found in the documents."
- You may provide general guidance but clearly label it as general information (not from documents).

Your role is to provide accurate, document-based information to help users find answers to their questions.
""".strip()


def init_agent(vectorstore_tuple, repo_owner: str, repo_name: str, model: str = "openai:gpt-4o-mini"):
    """
    Initialize the pydantic-ai agent with search tools.
    From notebook cell 47.
    
    Args:
        vectorstore_tuple: Tuple of (vectorstore, table) from ingest.index_data()
        repo_owner: Repository owner (for context)
        repo_name: Repository name (for context)
        model: Model identifier (default: "openai:gpt-4o-mini")
        
    Returns:
        Configured pydantic-ai Agent
    """
    # Unpack the vectorstore tuple
    vectorstore, table = vectorstore_tuple
    
    # Set up the search tools with the vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    search_tools.set_search_index(vectorstore, table, embeddings)
    
    # Create system prompt
    system_prompt = create_system_prompt(repo_owner, repo_name)
    
    # Create the agent with the text_search tool
    # From notebook cell 47
    agent = Agent(
        model,
        name="document_agent",
        system_prompt=system_prompt,
        tools=[search_tools.text_search],
    )
    
    print(f"Agent initialized with model: {model}")
    return agent