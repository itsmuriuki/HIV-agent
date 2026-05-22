"""
Search agent module using pydantic-ai.
Adapted from the Kenya ARV Guidelines notebook (cells 43, 47, 67).
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.gemini import GeminiModel
import search_tools
import ingest


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
    
    # Create system prompt
    system_prompt = create_system_prompt(repo_owner, repo_name)
    
    model_for_agent = model
    if isinstance(model, str) and model.startswith("deepseek:"):
        # Use pydantic-ai's built-in DeepSeek provider (OpenAI-compatible API).
        model_name = model.split(":", 1)[1]
        model_for_agent = OpenAIChatModel(model_name, provider="deepseek")
    elif isinstance(model, str) and model.startswith("gemini:"):
        # Use pydantic-ai's built-in Gemini model (Google Generative Language API).
        model_name = model.split(":", 1)[1]
        model_for_agent = GeminiModel(model_name, provider="google-gla")

    # Build a tool bound to THIS vectorstore (no globals).
    text_search_tool = search_tools.build_text_search(vectorstore, k=5)

    # Create the agent with the text_search tool
    # From notebook cell 47
    agent = Agent(
        model_for_agent,
        name="document_agent",
        system_prompt=system_prompt,
        tools=[text_search_tool],
    )
    
    print(f"Agent initialized with model: {model}")
    return agent