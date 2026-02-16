"""
Data ingestion module for PDF documents.
Adapted directly from the Kenya ARV Guidelines notebook (cells 1, 3, 11, 13, 21).
"""

import os
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document
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
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
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
    table_name = "documents"
    
    # Initialize embeddings model (from notebook cell 20)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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