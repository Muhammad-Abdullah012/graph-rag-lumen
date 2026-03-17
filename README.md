# Graph RAG Lumen

A document Q&A system that processes PDF files using Mistral OCR, stores structured page data in a Neo4j knowledge graph, and answers questions using locally hosted LLMs via Ollama with semantic vector search.

---

## Architecture

```
PDF Upload
    │
    ▼
Mistral OCR API
(page markdown + images + tables)
    │
    ▼
Neo4j Graph
  [Document] ──CONTAINS──► [Page] ──NEXT_PAGE──► [Page] ──► ...
                              │
                         embedding (bge-m3)
                         markdown (with tables inlined)
                         images, header, footer, dimensions
    │
    ▼
Vector Search (HNSW) + Fulltext Index fallback (Lucene)
    │
    ▼
Ollama LLM (mistral-nemo / any chat model)
    │
    ▼
Streamed Answer + Source Attribution
```

### Services

| Service | Image | Host Port | Purpose |
|---------|-------|-----------|---------|
| `backend` | Custom (FastAPI) | `9000` | API server, OCR pipeline, graph builder, Q&A |
| `frontend` | Custom (React) | `3006` | Web UI |
| `nginx` | nginx:alpine | `8089` | Reverse proxy + static document serving |
| `neo4j` | neo4j:5.26.1 | `8475` (HTTP), `8688` (Bolt) | Graph database + vector index |

Ollama runs externally (not managed by this Compose stack).

---

## Prerequisites

- Docker with Compose v2 (`docker compose`)
- NVIDIA GPU with CUDA (recommended — used by backend and Ollama)
- **Ollama** running with:
  - An LLM model, e.g. `mistral-nemo:12b`
  - An embedding model: `bge-m3:latest`
- **Mistral API key** (for PDF OCR via `mistral-ocr-latest`)

---

## Quick Start

```bash
./QUICKSTART.sh
```

The script will:
1. Create `.env` from `.env.example`
2. Create required directories
3. Build and start all Docker services
4. Wait for the backend to be healthy
5. Print access URLs

---

## Manual Setup

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — required values:

```bash
# Mistral OCR (get key at console.mistral.ai)
MISTRAL_API_KEY=your_key_here

# Ollama (must be reachable from inside Docker)
OLLAMA_BASE_URL=http://your-ollama-host:11434
OLLAMA_LLM_MODEL=mistral-nemo:12b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest

# Neo4j credentials (must match .env.neo)
NEO4J_PASSWORD=neo4jpassword
```

Also configure `.env.neo` for Neo4j container settings (auth, advertised addresses).

### 2. Create directories

```bash
mkdir -p documents logs backend/json backend/images
```

### 3. Start services

```bash
docker compose up -d --build
```

### 4. Access

| Interface | URL |
|-----------|-----|
| Web UI | http://localhost:8089 |
| API Docs (Swagger) | http://localhost:9000/docs |
| Health Check | http://localhost:9000/api/health/ |
| Neo4j Browser | http://localhost:8475 |

---

## Usage

### Upload & Process Documents

1. Open the web UI → **Dateien** tab
2. Drag & drop or select PDF files (max 50 MB each)
3. Click **Process All Documents** to run OCR and build the graph
4. Processing runs in the background — the status updates automatically

### Ask Questions

1. Switch to the **Chat** tab
2. Type your question (German and English both supported)
3. The system retrieves the most relevant pages and streams the answer
4. Sources show document name and page number

---

## Configuration Reference

All settings are loaded from environment variables. Defaults are in `config/settings.py`.

### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://neo4j:7687` | Bolt connection URI |
| `NEO4J_USERNAME` | `neo4j` | Username |
| `NEO4J_PASSWORD` | `password` | Password |
| `NEO4J_DATABASE` | `neo4j` | Database name |

### Ollama

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Chat model for Q&A |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `OLLAMA_NUM_CTX` | `16384` | LLM context window (tokens) |

### Mistral OCR

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | *(required)* | Mistral API key |
| `BATCH_POLL_INTERVAL` | `5` | Seconds between batch status checks |

### File Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENTS_PATH` | `/tmp/documents` | Where PDFs are stored |
| `JSON_OUTPUT_PATH` | `/backend/json` | OCR result JSON files |
| `IMAGES_PATH` | `/backend/images` | Extracted page images |
| `DEBUG_LOG_PATH` | `/app/debug_raw.jsonl` | Debug log (prompt + answer per query) |

### Retrieval & Search

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `5` | Number of pages to retrieve per query |
| `VECTOR_INDEX_NAME` | `page_embeddings` | Neo4j vector index name |
| `FULLTEXT_INDEX_NAME` | `page_fulltext` | Neo4j fulltext index name |

### Upload Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UPLOAD_SIZE` | `52428800` | Max file size in bytes (50 MB) |
| `PDF_EXTRACT_TIMEOUT` | `300` | OCR timeout in seconds |
| `GRAPH_BUILD_TIMEOUT` | `600` | Graph build timeout in seconds |

---

## API Reference

### Health

```
GET /api/health/
→ { status, neo4j, ollama, graph_loaded }
```

### Documents

```
GET  /api/documents
→ [ { document_id, filename, size_mb, upload_date } ]

POST /api/documents/upload
     multipart: files[]
→ [ { document_id, filename, status } ]

POST /api/documents/process-all
→ { message: "Processing started" }

GET  /api/documents/status/{document_id}
→ { status, message }   # status: pending | processing | completed | failed
```

### Q&A

```
POST /api/qa/stream
     { "question": "...", "top_k": 5 }
→ Server-Sent Events stream:
     data: { type: "status",  message: "..." }
     data: { type: "token",   content: "..." }
     data: { type: "done",    answer: "...", sources: [...] }
```

Each source item: `{ document, page_number, chapter, preview }`.

---

## Graph Schema

```
(:Document {
  id, name, source_file, url, page_count, processed_at
})

(:Page {
  id,           # {document_id}_p{index}
  document_id,
  page_number,  # 0-based index
  markdown,     # full page text with table placeholders
  tables_json,  # JSON array of { id, content (markdown table) }
  images_json,  # JSON array of { id, file_path, bbox coords }
  hyperlinks_json,
  header, footer,
  dimensions_json,
  embedding     # float[] from bge-m3
})

(:Page)-[:BELONGS_TO]->(:Document)
(:Page)-[:NEXT_PAGE]->(:Page)
```

**Indexes:**
- Vector index on `Page.embedding` (HNSW, cosine similarity)
- Fulltext index on `Page.markdown` (Lucene, fallback keyword search)
- Unique constraints on `Document.id` and `Page.id`

---

## Processing Pipeline

```
1. Upload PDF  →  saved to DOCUMENTS_PATH with UUID prefix
2. OCR         →  Mistral batch API (mistral-ocr-latest)
                  pages: markdown + images + tables + header/footer
                  images decoded and saved to IMAGES_PATH
                  result cached as JSON in JSON_OUTPUT_PATH
3. Graph Build →  Document node created/merged
                  Page nodes created with embeddings (bge-m3)
                  NEXT_PAGE links chained in order
4. Done        →  document status → "completed"
```

If a JSON file for the document already exists, OCR is skipped and the graph is rebuilt from cache.

---

## Retrieval Pipeline

```
1. Embed question  →  bge-m3 via Ollama
2. Vector search   →  db.index.vector.queryNodes (HNSW, top-k pages)
   Fallback        →  db.index.fulltext.queryNodes if vector fails
3. Build context   →  inline tables into markdown, truncate to OLLAMA_NUM_CTX budget
4. LLM generation  →  /api/chat with system + user messages, streamed
5. Return          →  token stream + source list
```

---

## Troubleshooting

**Neo4j database offline after restart**
```bash
docker compose down -v   # wipe volumes
docker compose up -d     # fresh start
```

**Ollama model not loading on GPU**

Check if another model is already occupying VRAM:
```bash
docker exec <ollama-container> ollama ps
docker exec <ollama-container> ollama stop <model-name>
```

**Context window too small (bad/truncated answers)**

Increase `OLLAMA_NUM_CTX` in `.env`. Default is `16384`. `mistral-nemo:12b` supports up to `128000`. Requires sufficient VRAM.

**mistralai import error**
```
ImportError: cannot import name 'Mistral' from 'mistralai'
```
Use `from mistralai.client import Mistral` — the package is a namespace package without `__init__.py` in some installations.

**OCR returns 402 (batch too large)**

Free Mistral tier limits batch size. Use single-document processing or upgrade your Mistral plan.

---

## Development

### Backend

```bash
cd /path/to/graph-rag-lumen
export PYTHONPATH=$(pwd)
pip install -r requirements.backend.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:9000 npm start
```

---

## Production Notes

- Set `ENVIRONMENT=production` in `.env`
- The stack integrates with **Traefik** via `traefik-net` external network
- Configure proper hostnames in `docker-compose.yaml` labels
- Neo4j is **Community Edition** — `START DATABASE` is not supported; restart the container to recover an offline database
- Ollama port `11434` should **not** be published (`ports:`) in production — use `expose:` only to prevent unauthorized external access
