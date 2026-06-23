"""
Ambulance Protocols API

RAG-powered Q&A for Kenya ambulance emergency dispatch and prehospital care protocols.

Run from the kenya-hiv-cdss/ directory:
    uvicorn api.main:app --reload
"""

import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Add app/ to sys.path so ingest and search_tools are importable
API_DIR = Path(__file__).resolve().parent
PROJECT_DIR = API_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "app"))

import ingest
import search_tools

from .llm_client import LLMClientError, LLMTarget, complete_chat
from .schemas import DispatchRequest, ProtocolResponse

load_dotenv(PROJECT_DIR / ".env")

# Strip provider prefix from MODEL env var (e.g. "openai:gpt-4o-mini" → "gpt-4o-mini")
_raw_model = os.getenv("MODEL", "openai:gpt-4o-mini")
MODEL_NAME = _raw_model.split(":", 1)[-1]
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
API_KEY = os.getenv("OPENAI_API_KEY", "")

AMBULANCE_PDF_PATHS = [
    str(PROJECT_DIR / "Ambulensi Emergency Medical Dispatch Protocols (1).pdf"),
    str(PROJECT_DIR / "Ambulensi Prehospital Emergency Care Clinical Protocols (1).pdf"),
]
DB_PATH = str(PROJECT_DIR / "lancedb" / f"ambulance_{ingest.embeddings_backend_name()}")

_text_search = None
_target: LLMTarget | None = None

SYSTEM_PROMPT = """You are a prehospital emergency dispatch assistant for the Kenya Ambulance Service.

You will be given:
1. An incident description with patient details.
2. Relevant sections retrieved from the official Ambulensi emergency dispatch and prehospital care protocol documents.

Your task:
- Return a JSON object with a single key "protocols" containing an ordered list of actionable prehospital steps.
- Each item must be one clear, actionable step a paramedic or dispatcher can act on immediately.
- Ground every step strictly in the retrieved document sections provided. Do NOT add steps from general knowledge not present in the documents.
- Cite the source document and page for each step using the format [document][page] at the end of the step.
- If the retrieved sections do not contain enough information for a step, omit that step rather than inventing it.

Return ONLY valid JSON — no markdown fences, no explanation outside the JSON object.

Example output format:
{"protocols": ["Ensure scene safety and don PPE [Ambulensi Dispatch Protocols][12]", "Assess airway, breathing, and circulation (ABCs) [Ambulensi Prehospital Protocols][34]"]}"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _text_search, _target
    print(f"Loading ambulance protocols index from: {DB_PATH}")
    vectorstore, _ = ingest.index_pdfs(AMBULANCE_PDF_PATHS, db_path=DB_PATH)
    _text_search = search_tools.build_text_search(vectorstore, k=5)
    _target = LLMTarget(
        name="LLM",
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    print(f"Ambulance agent ready. Model: {MODEL_NAME}, Base URL: {BASE_URL}")
    yield


app = FastAPI(
    title="Ambulance Protocols API",
    description="Q&A API grounded in Kenya ambulance emergency dispatch and prehospital care protocol documents.",
    version="0.1.0",
    lifespan=lifespan,
)


def _build_query(request: DispatchRequest) -> str:
    """Assemble a rich query string from the structured dispatch payload."""
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


def _build_messages(query: str, context_chunks: list[str]) -> list[dict]:
    """Build the messages list for the chat completions call."""
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant sections found."
    user_content = (
        f"Incident:\n{query}"
        f"\n\nRetrieved protocol sections from the Ambulensi documents:\n{context}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_protocols(raw: str) -> list[str]:
    """Extract the protocols list from the LLM's JSON response.

    Strips markdown code fences in case the model wraps its output in ```json ... ```.
    """
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    data = json.loads(cleaned)
    protocols = data.get("protocols", [])
    if not isinstance(protocols, list):
        raise ValueError(f"Expected 'protocols' to be a list, got {type(protocols)}")
    return [str(p) for p in protocols]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_ready": _target is not None,
        "model": MODEL_NAME,
        "base_url": BASE_URL,
    }


@app.post("/em_protocols_rag", response_model=ProtocolResponse)
async def em_protocols_rag(request: DispatchRequest):
    if _target is None or _text_search is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialized — try again in a moment.")

    # Step 1 — build the query from the structured dispatch fields
    query = _build_query(request)

    # Step 2 — retrieve the most relevant protocol chunks from LanceDB
    context_chunks = _text_search(query)

    # Step 3 — build the messages list (system prompt + retrieved context + query)
    messages = _build_messages(query, context_chunks)

    # Step 4 — call the LLM directly via HTTP
    try:
        raw, _usage = await complete_chat(_target, messages)
    except LLMClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 5 — parse the JSON response into the protocol list
    try:
        protocols = _parse_protocols(raw)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not parse LLM response: {e}. Raw: {raw[:300]}",
        )

    return ProtocolResponse(protocols=protocols)
