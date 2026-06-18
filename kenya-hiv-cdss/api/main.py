"""
Ambulance Protocols API

RAG-powered Q&A for Kenya ambulance emergency dispatch and prehospital care protocols.

Run from the kenya-hiv-cdss/ directory:
    uvicorn api.main:app --reload
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException

# Add app/ to sys.path so ingest, search_agent, etc. are importable
API_DIR = Path(__file__).resolve().parent
PROJECT_DIR = API_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "app"))

import ingest
import search_agent

from .schemas import DispatchRequest, ProtocolResponse, ProtocolsOutput

load_dotenv(PROJECT_DIR / ".env")

MODEL = os.getenv("MODEL", "openai:gpt-4o-mini")

AMBULANCE_PDF_PATHS = [
    str(PROJECT_DIR / "Ambulensi Emergency Medical Dispatch Protocols (1).pdf"),
    str(PROJECT_DIR / "Ambulensi Prehospital Emergency Care Clinical Protocols (1).pdf"),
]
DB_PATH = str(PROJECT_DIR / "lancedb" / f"ambulance_{ingest.embeddings_backend_name()}")

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    print(f"Loading ambulance protocols index from: {DB_PATH}")
    index = ingest.index_pdfs(AMBULANCE_PDF_PATHS, db_path=DB_PATH)
    _agent = search_agent.init_agent(
        index, "Ambulance", "Emergency-Protocols", model=MODEL, output_type=ProtocolsOutput
    )
    print("Ambulance agent ready.")
    yield


app = FastAPI(
    title="Ambulance Protocols API",
    description="Q&A API grounded in Kenya ambulance emergency dispatch and prehospital care protocol documents.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_agent():
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialized — try again in a moment.")
    return _agent


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": _agent is not None, "model": MODEL}


def _build_query(request: DispatchRequest) -> str:
    """Build a rich query string from the dispatch payload for the RAG agent."""
    parts = [request.incidentInfo.description]

    p = request.patientInfo
    if p:
        if p.sex:
            parts.append(f"Sex: {p.sex}")
        if p.ageGroup:
            parts.append(f"Age group: {p.ageGroup}")
        if p.approxAge:
            parts.append(f"Approximate age: {p.approxAge}")

    i = request.incidentInfo
    if i.consciousness:
        parts.append(f"Consciousness: {i.consciousness}")
    if i.breathing:
        parts.append(f"Breathing: {i.breathing}")
    if i.activelyBleeding:
        parts.append("Actively bleeding: yes")
    if i.medicalHistory:
        parts.append(f"Medical history: {i.medicalHistory}")

    return ". ".join(parts)


@app.post("/em_protocols_rag", response_model=ProtocolResponse)
async def em_protocols_rag(request: DispatchRequest, agent=Depends(get_agent)):
    query_text = _build_query(request)
    try:
        result = await agent.run(query_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ProtocolResponse(protocols=result.output.protocols)
