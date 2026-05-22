"""
Data ingestion module for PDF documents.
Adapted directly from the Kenya ARV Guidelines notebook (cells 1, 3, 11, 13, 21).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401
    from langchain_openai import OpenAIEmbeddings  # noqa: F401
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
import lancedb
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


TABLE_NAME = "documents"

# Directory containing this module (kenya-hiv-cdss/app/)
_APP_DIR = Path(__file__).resolve().parent
# Project root (kenya-hiv-cdss/)
_PROJECT_DIR = _APP_DIR.parent


def _resolve_pdf_path(pdf_path: str) -> str:
    """
    Resolve a PDF path robustly across different working directories.

    Resolution order:
    - as provided (absolute or relative to CWD)
    - relative to project root (kenya-hiv-cdss/)
    - relative to app dir (kenya-hiv-cdss/app/)
    """
    p = Path(pdf_path)
    if p.is_file():
        return str(p)

    proj_path = _PROJECT_DIR / pdf_path
    if proj_path.is_file():
        return str(proj_path)

    app_path = _APP_DIR / p.name
    if app_path.is_file():
        return str(app_path)

    raise FileNotFoundError(
        f"PDF not found: {pdf_path} (also tried {proj_path} and {app_path})."
    )


def get_embeddings():
    """
    Select embeddings backend.

    By default we use local HuggingFace embeddings to avoid unexpected OpenAI quota/billing issues
    during indexing. To force OpenAI embeddings set:

    - `EMBEDDINGS_BACKEND=openai`
    """
    backend = (os.environ.get("EMBEDDINGS_BACKEND") or "hf").strip().lower()

    if backend in {"openai", "oai"}:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Lazy import to avoid importing torch/sentence-transformers unless needed.
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Local embeddings backend could not be loaded. "
            "Set OPENAI_API_KEY to use OpenAI embeddings instead."
        ) from e

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def embeddings_backend_name() -> str:
    """A stable name for the active embeddings backend (used for index folder naming)."""
    backend = (os.environ.get("EMBEDDINGS_BACKEND") or "hf").strip().lower()
    if backend in {"openai", "oai"}:
        return "openai"
    return "hf"


def intelligent_chunking(text: str) -> List[str]:
    """
    Split text into intelligent chunks using RecursiveCharacterTextSplitter.
    From notebook cell 11.
    
    Args:
        text: Text to chunk
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def process_doc(doc: Document) -> List[Dict[str, Any]]:
    """
    Splits a single document into sections and returns a list of section dicts.
    From notebook cell 13.
    
    Args:
        doc: LangChain Document object
        
    Returns:
        List of chunked documents with metadata
    """
    doc_content = doc.page_content
    doc_metadata = doc.metadata
    sections = intelligent_chunking(doc_content)
    return [{**doc_metadata, 'section': section} for section in sections]


def load_and_chunk_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Load PDF and chunk documents in parallel.
    Adapted from notebook cells 1, 3, 13.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of chunked documents
    """
    # Load PDF (from notebook cell 1)
    resolved = _resolve_pdf_path(pdf_path)
    print(f"Loading PDF: {resolved}")
    loader = PyPDFLoader(resolved)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # Process documents in parallel with progress bar (from notebook cell 13)
    guides_chunks = []
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_doc, doc) for doc in documents]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing docs"):
            guides_chunks.extend(future.result())
    
    print(f"Total chunks created: {len(guides_chunks)}")
    return guides_chunks


def load_and_chunk_pdfs(pdf_paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load and chunk multiple PDFs, combining all chunks into one list."""
    all_chunks: List[Dict[str, Any]] = []
    for pdf_path in pdf_paths:
        chunks = load_and_chunk_pdf(pdf_path)
        # Ensure we retain a stable doc identifier in metadata
        for c in chunks:
            c.setdefault("document_title", Path(c.get("source", pdf_path)).name)
        all_chunks.extend(chunks)
    return all_chunks


def create_vector_index(guides_chunks: List[Dict[str, Any]], db_path: str = "./lancedb") -> tuple:
    """
    Create a LanceDB vector index from chunks.
    From notebook cell 21.
    
    Args:
        guides_chunks: List of document chunks
        db_path: Path to LanceDB database
        
    Returns:
        Tuple of (vectorstore, table)
    """
    table_name = TABLE_NAME
    embeddings = get_embeddings()
    
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    
    # Drop existing table if it exists (for fresh start)
    if table_name in db.table_names():
        print(f"Dropping existing table '{table_name}'...")
        db.drop_table(table_name)
    
    print("Creating new vectorstore...")
    
    # Extract texts and metadatas
    texts = [chunk["section"] for chunk in guides_chunks]
    metadatas = guides_chunks
    
    # Generate embeddings manually
    print("Generating embeddings...")
    vectors = embeddings.embed_documents(texts)
    
    # Create the table manually with proper schema
    data = []
    for i, (text, metadata, vector) in enumerate(zip(texts, metadatas, vectors)):
        data.append({
            "text": text,
            "vector": vector,
            "id": str(i),
            "source": metadata.get("source", ""),
            "page": metadata.get("page", 0)
        })
    
    # Create table
    table = db.create_table(table_name, data=data, mode="overwrite")
    print(f"✓ Created table with {len(data)} records")
    
    # Now create the LanceDB vectorstore wrapper
    vectorstore = LanceDB(
        connection=table,
        embedding=embeddings
    )
    
    print("✓ Vectorstore created successfully!")
    
    return vectorstore, table


def index_data(pdf_path: str, db_path: str = "./lancedb") -> tuple:
    """
    Main function to index data from PDF.
    
    Args:
        pdf_path: Path to PDF file
        db_path: Path to LanceDB database
        
    Returns:
        Tuple of (vectorstore, table)
    """
    print(f"Processing PDF: {pdf_path}")
    guides_chunks = load_and_chunk_pdf(pdf_path)
    
    print("\nBuilding vector index...")
    vectorstore, table = create_vector_index(guides_chunks, db_path)
    
    return vectorstore, table


def _load_existing_index(db_path: str) -> Optional[Tuple]:
    """
    If the LanceDB db at db_path already has the documents table, open it and
    return (vectorstore, table). Otherwise return None so caller does full index.
    """
    try:
        db = lancedb.connect(db_path)
        if TABLE_NAME not in db.table_names():
            return None
        table = db.open_table(TABLE_NAME)
        embeddings = get_embeddings()
        vectorstore = LanceDB(connection=table, embedding=embeddings)
        print("✓ Loaded existing vector index (skipping PDF processing)")
        return vectorstore, table
    except Exception as e:
        print(f"Could not load existing index: {e}")
        return None


def index_pdfs(pdf_paths: Sequence[str], db_path: str) -> tuple:
    """
    Index multiple PDFs into a single LanceDB database, reusing an existing index if present.
    """
    existing = _load_existing_index(db_path)
    if existing is not None:
        return existing

    print(f"Processing PDFs: {list(pdf_paths)}")
    guides_chunks = load_and_chunk_pdfs(pdf_paths)
    print("\nBuilding vector index...")
    vectorstore, table = create_vector_index(guides_chunks, db_path)
    return vectorstore, table