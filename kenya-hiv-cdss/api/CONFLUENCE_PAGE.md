# Ambulance Protocols RAG Microservice

**Service Name:** `ambulance-protocols-api`  
**Version:** 0.2.0  
**Owner:** Digital Health Tribe — CDSS Squad  
**Status:** In Development  
**Repository:** `gitlab.safaricom.co.ke/digital-health/cdss-rag` → branch `feature/ambulance-cdss-api`

---

## 1. What This Service Does

The Ambulance Protocols API is a **RAG (Retrieval-Augmented Generation) microservice** that generates step-by-step prehospital and emergency dispatch protocols for a given incident.

When a dispatcher fills in the emergency dispatch form on the SHA Emergency Portal, this service receives the incident details and returns a list of actionable clinical protocols grounded in two official Kenya ambulance documents:

- **Ambulensi Emergency Medical Dispatch Protocols**
- **Ambulensi Prehospital Emergency Care Clinical Protocols**

The service does **not** generate protocols from general AI knowledge — every protocol returned is retrieved from and grounded in these documents.

---

## 2. How It Fits Into the TaifaCare System

```
Dispatcher fills form on SHA Portal (emergency.sha.go.ke)
        │
        ▼
   webListener (DOM tracker)
        │  sends full dispatch JSON
        ▼
   TaifaCare Backend (FastAPI)
        │  POST /em_protocols_rag
        ▼
   ┌─────────────────────────────────┐
   │  Ambulance Protocols RAG API    │  ◄── THIS SERVICE
   │  (this microservice)            │
   └─────────────────────────────────┘
        │  { "protocols": [...] }
        ▼
   TaifaCare Backend
        │
        ▼
   React HUD displayed to dispatcher
```

The TaifaCare backend calls this service directly. This service is not exposed to the browser.

---

## 3. API Reference

### Base URL
```
http://<host>:8000
```

---

### GET /health

Check if the service is running and the agent is ready.

**Request**
```
GET /health
```

**Response — 200 OK**
```json
{
  "status": "ok",
  "agent_ready": true,
  "model": "gpt-4o-mini",
  "base_url": "https://api.openai.com/v1"
}
```

| Field | Description |
|---|---|
| `status` | Always `"ok"` if the server is reachable |
| `agent_ready` | `true` once the vector index and LLM client have finished loading |
| `model` | The LLM model name currently in use |
| `base_url` | The LLM provider endpoint currently in use |

> **Note:** On first startup, `agent_ready` will be `false` for ~60 seconds while the PDF index is being built. Call `/health` first before sending queries.

---

### POST /em_protocols_rag

Accepts the full dispatch JSON payload and returns a list of prehospital protocols relevant to the incident.

**Request**

```
POST /em_protocols_rag
Content-Type: application/json
```

**Request Body**

The service accepts the full dispatch JSON. The most critical field is `incidentInfo.description`. All other fields are used as supplementary clinical context if filled in.

```json
{
  "dispatchId": "a2e4b148-d913-4c63-98ba-10386c0c8a84",
  "patientInfo": {
    "ageGroup": "adult",
    "approxAge": "55",
    "sex": "male"
  },
  "incidentInfo": {
    "description": "55-year-old male with a sudden severe headache",
    "priority": "high",
    "consciousness": "alert",
    "breathing": "normal",
    "activelyBleeding": false,
    "medicalHistory": "hypertension"
  }
}
```

**Request Fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `dispatchId` | string | Yes | Unique ID for the dispatch incident |
| `incidentInfo.description` | string | Yes | Free-text incident description — **primary input to the RAG system** |
| `patientInfo.ageGroup` | string | No | e.g. `"adult"`, `"child"` |
| `patientInfo.sex` | string | No | e.g. `"male"`, `"female"` |
| `patientInfo.approxAge` | string | No | Approximate age |
| `incidentInfo.consciousness` | string | No | e.g. `"alert"`, `"unconscious"` |
| `incidentInfo.breathing` | string | No | e.g. `"normal"`, `"laboured"` |
| `incidentInfo.activelyBleeding` | boolean | No | `true` / `false` |
| `incidentInfo.medicalHistory` | string | No | Relevant medical history |

> All other fields in the full dispatch payload (callerInfo, logistics, rawForm, interactionMetadata, etc.) are accepted but **ignored** — they do not affect the response.

---

**Response — 200 OK**

```json
{
  "protocols": [
    "Ensure scene safety and use appropriate personal protective equipment (PPE).",
    "Assess airway, breathing, and circulation (ABCs).",
    "Perform a rapid neurological assessment (AVPU scale).",
    "Monitor blood pressure — hypertensive emergency should be managed with caution.",
    "Position the patient with head elevated at 30 degrees if no spinal injury is suspected.",
    "Establish IV access and monitor vital signs continuously.",
    "Transport urgently to the nearest Level 5/6 facility with neurosurgical capability."
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `protocols` | array of strings | Ordered list of actionable prehospital/dispatch steps grounded in the ambulance protocol documents |

---

**Response — 503 Service Unavailable**

Returned if the request arrives before the agent has finished loading on startup.

```json
{
  "detail": "Agent not yet initialized — try again in a moment."
}
```

**Response — 500 Internal Server Error**

Returned if the LLM call fails (e.g. invalid API key, quota exceeded).

```json
{
  "detail": "status_code: 401 ..."
}
```

---

## 4. How the RAG System Works Internally

```
1. incidentInfo.description + any filled clinical fields
           │
           ▼
   Query builder assembles a rich text query
   e.g. "55-year-old male with severe headache. Sex: male. Medical history: hypertension"
           │
           ▼
   HuggingFace embeddings model converts query to a vector
   (sentence-transformers/all-MiniLM-L6-v2 — runs locally, no external API)
           │
           ▼
   LanceDB vector search finds the 5 most relevant chunks
   from the two ambulance protocol PDFs
           │
           ▼
   Retrieved chunks + query sent to LLMClient (see Section 5)
   Direct HTTP POST to OpenAI-compatible endpoint — no LangChain wrapper
           │
           ▼
   LLM returns structured { "protocols": [...] }
   grounded strictly in the retrieved document sections
```

**Key guarantee:** The LLM is instructed to only use information found in the documents. It will not invent protocols.

---

## 5. LLM Client (`api/llm_client.py`)

The microservice uses a purpose-built **LLM client** (`llm_client.py`) instead of the LangChain LLM wrappers. It makes direct HTTP calls to any OpenAI-compatible endpoint using `httpx`, giving us full control over the request and no indirect dependency chain.

### Classes

#### `LLMTarget`

Holds the configuration for a specific LLM provider endpoint. Constructed once at startup from environment variables and reused for every request.

```python
@dataclass
class LLMTarget:
    name: str        # Human-readable label (e.g. "openai", "bedrock")
    model: str       # Model name (e.g. "gpt-4o-mini")
    api_key: str     # API key for the provider
    base_url: str    # Provider endpoint. Default: "https://api.openai.com/v1"
```

`LLMTarget.chat_completions_url()` returns `{base_url}/chat/completions`.

#### `UsageStats`

Token usage returned alongside every LLM response.

```python
@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

#### `LLMClientError`

Raised when the HTTP call fails or the response cannot be parsed. Wraps the original HTTP status and body.

---

### Function: `complete_chat`

```python
async def complete_chat(
    target: LLMTarget,
    messages: list[dict],
    *,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> tuple[str, UsageStats]:
```

Sends a `messages` array to `target.chat_completions_url()` and returns `(response_text, usage_stats)`.

- Fully async (`httpx.AsyncClient`)
- Raises `LLMClientError` on any non-200 HTTP status or malformed JSON
- Used directly in the `/em_protocols_rag` endpoint

### Why a custom client instead of LangChain?

| Consideration | LangChain wrapper | `llm_client.py` |
|---|---|---|
| Dependency footprint | Heavy (pulls in full LangChain) | `httpx` only |
| Provider compatibility | Depends on specific adapters | Any OpenAI-compatible URL |
| Streaming control | Abstracted away | Direct |
| Error visibility | Wrapped exceptions | Raw HTTP status + body |

---

## 6. Environment Variables

These must be set at deployment time. **Never commit these to git.**

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | API key for the LLM provider |
| `OPENAI_BASE_URL` | Yes (for Bedrock) | Custom base URL if using AWS Bedrock or another OpenAI-compatible endpoint. Default: `https://api.openai.com/v1` |
| `MODEL` | No | LLM model to use in `provider:model` format. Default: `openai:gpt-4o-mini`. Other options: `gemini:gemini-2.0-flash`, `deepseek:deepseek-chat` |
| `EMBEDDINGS_BACKEND` | No | `hf` (default, local HuggingFace) or `openai` (uses OpenAI embeddings API) |
| `EMBEDDINGS_BATCH_SIZE` | No | Batch size for OpenAI embeddings. Default: `12` |
| `EMBEDDINGS_BATCH_PAUSE` | No | Pause in seconds between OpenAI embedding batches. Default: `1.5` |

---

## 7. Deployment (For DevOps)

The service is fully containerised. The Dockerfile is at `kenya-hiv-cdss/Dockerfile` in the repository.

### Build the image
```bash
docker build -t ambulance-protocols-api .
```

### Run the container
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e OPENAI_BASE_URL=your-bedrock-url \
  -v ambulance-lancedb:/app/lancedb \
  ambulance-protocols-api
```

The `-v` flag mounts a persistent volume for the LanceDB index. Without it, the index is rebuilt from the PDFs every time the container starts (~60 seconds). With it, the index is built once and reused on subsequent starts (~5 seconds).

### docker-compose snippet
Add this to the TaifaCare `docker-compose.yml`:

```yaml
ambulance-rag:
  build:
    context: ./cdss-rag/kenya-hiv-cdss
    dockerfile: Dockerfile
  ports:
    - "8000:8000"
  environment:
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - OPENAI_BASE_URL=${OPENAI_BASE_URL}
  volumes:
    - ambulance-lancedb:/app/lancedb
  restart: unless-stopped

volumes:
  ambulance-lancedb:
```

The TaifaCare backend then calls this service at:
```
http://ambulance-rag:8000/em_protocols_rag
```

---

## 8. Running Locally (For Developers)

```bash
# 1. Clone the repo
git clone https://gitlab.safaricom.co.ke/digital-health/cdss-rag.git
cd cdss-rag

# 2. Checkout the feature branch
git checkout feature/ambulance-cdss-api

# 3. Create and activate a virtual environment
#    Use a short path to avoid Windows long-path issues
python -m venv C:/venvs/cdss-rag
source C:/venvs/cdss-rag/Scripts/activate   # Windows Git Bash
# or
source C:/venvs/cdss-rag/bin/activate       # Mac/Linux

# 4. Install dependencies
cd kenya-hiv-cdss
pip install -r api/requirements.txt

# 5. Create .env with your API key
echo 'OPENAI_API_KEY=your-key' > .env
echo 'OPENAI_BASE_URL=your-bedrock-url' >> .env

# 6. Start the server
uvicorn api.main:app --reload
```

Interactive docs available at: `http://localhost:8000/docs`

A test HTTP file (`ambulance_protocols_api.http`) is included at the repo root for use with the VS Code REST Client extension.

---

## 9. Knowledge Base (Source Documents)

| Document | Pages | Description |
|---|---|---|
| Ambulensi Emergency Medical Dispatch Protocols | 84 pages → 204 chunks | Protocols for emergency call-taking and dispatch decisions |
| Ambulensi Prehospital Emergency Care Clinical Protocols | 224 pages → 403 chunks | Clinical protocols for paramedics/EMTs on scene |

Total indexed: **607 chunks** stored in LanceDB.

Chunks are produced by `RecursiveCharacterTextSplitter` with `chunk_size=800`, `chunk_overlap=100`.

---

## 10. Module Overview

| File | Purpose |
|---|---|
| `api/main.py` | FastAPI app, lifespan startup, `/health` and `/em_protocols_rag` endpoints |
| `api/llm_client.py` | `LLMTarget`, `complete_chat()` — direct HTTP LLM calls via `httpx` |
| `api/schemas.py` | Pydantic models: `DispatchRequest`, `PatientInfo`, `IncidentInfo`, `ProtocolResponse` |
| `app/ingest.py` | PDF loading, chunking, LanceDB indexing, embeddings backend selection |
| `app/search_tools.py` | `build_text_search()` — stateless vector search tool factory |
| `app/search_agent.py` | `init_agent()` — pydantic-ai agent used by the Streamlit app |
| `app/ui_common.py` | Shared Streamlit utilities: env loading, async loop, page init helpers |
| `app/app.py` | Streamlit entry point for the developer-facing Q&A UI |

---

## 11. Known Issues & Limitations

| Issue | Status | Notes |
|---|---|---|
| AWS Bedrock base URL not yet configured | Pending | Waiting on correct `OPENAI_BASE_URL` from infra team |
| HuggingFace symlink warning on Windows | Non-blocking | Warning only, does not affect functionality |
| LangChain `HuggingFaceEmbeddings` deprecation warning | Non-blocking | Will be updated to `langchain-huggingface` in next iteration |
| First startup takes ~60s | By design | LanceDB index is built from PDFs; use a persistent volume to avoid on subsequent starts |
