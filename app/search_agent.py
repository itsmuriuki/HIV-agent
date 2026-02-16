"""
Search agent module using pydantic-ai.
Adapted from the Kenya ARV Guidelines notebook (cells 43, 47, 67).
"""

from pydantic_ai import Agent
import search_tools
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_system_prompt(repo_owner: str, repo_name: str) -> str:
    """
    Create the system prompt for the Kenya HIV clinical decision support agent.

    Args:
        repo_owner: Repository owner (for context)
        repo_name: Repository name (for context)

    Returns:
        System prompt string
    """
    return """
You are a clinical decision support assistant for the Kenya National HIV/AIDS treatment guidelines.

Before answering any clinical question, you MUST search the official ARV guideline materials for relevant information.

If the initial search does not return sufficient or precise information:
- Refine the query using clinical synonyms (e.g., regimen names, drug names, WHO stage, pregnancy, TB co-infection, CD4 thresholds).
- Perform multiple searches if necessary.
- Combine relevant retrieved sections before forming a response.

When relevant guideline content is found:
- Base your answer strictly and only on the retrieved guideline information.
- Do NOT introduce external medical knowledge.
- Provide clear, structured clinical output (e.g., Eligibility, Recommended Regimen, Dosing, Special Populations, Monitoring).
- Reference the relevant section or source when available.

If no relevant information is found after multiple searches:
- Clearly state: "This information was not found in the Kenya ARV guideline materials."
- Provide general best-practice guidance separately and clearly label it as general information.
- Avoid making definitive clinical recommendations outside the retrieved guidelines.

Safety Rules:
- Do not invent regimens, dosages, or thresholds.
- Do not assume missing patient details.
- If essential clinical variables are missing (e.g., age, pregnancy status, TB status, viral load, CD4 count), ask for clarification before answering.
- Clearly state any uncertainty or limitations.

Your role is to support clinicians with guideline-based information, not to replace clinical judgment.
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