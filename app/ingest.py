"""
Data ingestion module for PDF documents.
Adapted directly from the Kenya ARV Guidelines notebook (cells 1, 3, 11, 13, 21).
"""

import os

# Force CPU before any torch/sentence_transformers imports (avoids meta tensor error on Streamlit Cloud)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path

# Use a writeable dir for LanceDB config in restricted environments (e.g. Streamlit Cloud)
if not os.environ.get("LANCEDB_CONFIG_DIR"):
    _config_dir = Path.home() / ".config" / "lancedb"
    if not _config_dir.exists() or not os.access(_config_dir, os.W_OK):
        os.environ["LANCEDB_CONFIG_DIR"] = "/tmp"
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Directory containing this module (app/ when deployed)
_APP_DIR = Path(__file__).resolve().parent


def _resolve_pdf_path(pdf_path: str) -> str:
    """Resolve PDF path: use as-is if it exists, else try relative to app dir."""
    p = Path(pdf_path)
    if p.is_file():
        return str(p)
    # Try next to this module (e.g. app/Kenya-ARV-Guidelines-2022-Final-1.pdf)
    app_path = _APP_DIR / p.name
    if app_path.is_file():
        return str(app_path)
    raise FileNotFoundError(
        f"PDF not found: {pdf_path} or {app_path}. "
        "Ensure the guideline PDF is in the app directory or set the correct path."
    )
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    
    # Initialize embeddings model (from notebook cell 20)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    
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


TABLE_NAME = "documents"


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
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        vectorstore = LanceDB(connection=table, embedding=embeddings)
        print("✓ Loaded existing vector index (skipping PDF processing)")
        return vectorstore, table
    except Exception as e:
        print(f"Could not load existing index: {e}")
        return None


def index_data(pdf_path: str, db_path: str = "./lancedb") -> tuple:
    """
    Main function to index data from PDF. Reuses existing LanceDB index at db_path
    when present so reruns/new processes don't rebuild from PDF.
    
    Args:
        pdf_path: Path to PDF file
        db_path: Path to LanceDB database
        
    Returns:
        Tuple of (vectorstore, table)
    """
    existing = _load_existing_index(db_path)
    if existing is not None:
        return existing

    print(f"Processing PDF: {pdf_path}")
    guides_chunks = load_and_chunk_pdf(pdf_path)

    print("\nBuilding vector index...")
    vectorstore, table = create_vector_index(guides_chunks, db_path)

    return vectorstore, table