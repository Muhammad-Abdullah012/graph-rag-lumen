# Graph RAG System

A production-ready Knowledge Graph RAG (Retrieval-Augmented Generation) system that processes PDF documents locally using Docling, builds a knowledge graph with Neo4j, and answers questions using local LLMs (Ollama).

## Features

- 📄 **PDF Processing**: Extract text from PDFs with OCR, table structure detection, and code enrichment using Docling
- 🔗 **Knowledge Graph**: Automatically build knowledge graphs from extracted text using LLMGraphTransformer
- 🧠 **Vector Embeddings**: Generate and store embeddings using Nomic embed model
- 🤖 **Local LLMs**: Use locally hosted Ollama models (llama3.4 for graph building, llama3.2 for Q&A)
- ❓ **Q&A System**: Answer questions with source attribution and confidence scores
- 🌐 **Web UI**: Modern React frontend for document upload and Q&A
- 🐳 **Docker**: Production-ready Docker Compose setup
- 🇩🇪 **German Language**: Optimized for German language documents

## Architecture

```
┌─────────────────┐
│   React UI      │
└────────┬────────┘
         │
      nginx:8080
         │
    ┌────┴────────────────┐
    │                     │
┌───▼───────┐   ┌────────▼──────┐
│  Backend  │   │   Frontend    │
│FastAPI   │   │   (Static)    │
└───┬───────┘   └───────────────┘
    │
    ├────► Neo4j (Graph DB + Vector Index)
    ├────► Ollama (Local LLMs)
    └────► File System (PDFs)
```

### Components

1. **Backend (Python/FastAPI)**
   - Document upload and processing
   - PDF text extraction (Docling)
   - Text splitting (MarkdownHeaderTextSplitter)
   - Graph building (LLMGraphTransformer)
   - Q&A system with vector search

2. **Frontend (React)**
   - Document upload interface
   - Q&A chat interface
   - Source attribution display
   - Health status monitoring

3. **Database (Neo4j)**
   - Knowledge graph storage
   - Vector index for semantic search
   - Document metadata

4. **LLMs (Ollama)**
   - `llama3.4`: Graph transformation
   - `llama3.2`: Q&A
   - `nomic-embed-text:latest`: Embeddings

## Prerequisites

- Docker & Docker Compose
- Ollama running with models:
  - `llama3.4`
  - `llama3.2`
  - `nomic-embed-text:latest`
- Ngrok tunnel to Ollama (or local access if on same network)

## Quick Start

### 1. Setup Environment

```bash
cp .env.example .env
# Edit .env and update:
# - OLLAMA_BASE_URL
# - NEO4J_PASSWORD
```

### 2. Create Required Directories

```bash
mkdir -p documents logs
```

### 3. Start Services

```bash
docker-compose up -d
```

### 4. Access Application

- **Frontend**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

### 5. Upload Documents

1. Go to "Upload Documents" tab
2. Select a PDF file
3. Wait for processing to complete
4. Ask questions about your documents

## Configuration

### Environment Variables

See `.env.example` for all available options:

```bash
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password

# Ollama
OLLAMA_BASE_URL=https://your-ngrok-url/
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_GRAPH_MODEL=llama3.4

# Documents
DOCUMENTS_PATH=/documents
DOCUMENTS_BASE_URL=http://localhost:8080/documents
```

## API Endpoints

### Health Check
```
GET /api/health/
```

### Document Operations
```
POST /api/documents/upload
- Upload a PDF document
- Returns: { document_id, filename, status }

GET /api/documents/status/{document_id}
- Check processing status
- Returns: { status, message }
```

### Question Answering
```
POST /api/qa/ask
- Body: { "question": "...", "top_k": 5 }
- Returns: { answer, sources, confidence }
```

## Development

### Backend Development

```bash
cd backend
pip install -r ../requirements.backend.txt
export PYTHONPATH=/path/to/project
python -m uvicorn main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm start
```

## Processing Pipeline

1. **Upload**: User uploads PDF
2. **Extraction**: Docling extracts text with OCR, tables, code
3. **Splitting**: MarkdownHeaderTextSplitter splits by headers
4. **Embedding**: Generate vector embeddings for each chunk
5. **Graph Building**: LLMGraphTransformer creates entities/relationships
6. **Storage**: Store in Neo4j with vector index

## Query Pipeline

1. **Embedding**: Generate embedding for question
2. **Retrieval**: Vector similarity search in Neo4j
3. **Context**: Gather top-k chunks as context
4. **Generation**: LLM generates answer with context
5. **Attribution**: Return answer with source documents

## Performance Tuning

### For Large Documents
- Increase `pdf_extract_timeout` in `.env`
- Increase `graph_build_timeout` in `.env`
- Consider chunk_size and overlap in settings

### For Better Accuracy
- Use larger models (llama3.4, not 3.2)
- Increase `top_k` in Q&A requests
- Fine-tune chunk overlap settings

## Troubleshooting

### Neo4j Connection Failed
```
Check: NEO4J_URI and credentials
Wait: Neo4j container takes 30s+ to fully start
```

### Ollama Connection Failed
```
Check: OLLAMA_BASE_URL is correct and accessible
Verify: Ngrok tunnel is active
Test: curl https://your-ngrok-url/api/tags
```

### GPU Not Used
```
Set: OLLAMA environment variable device='auto'
Or: Manually set device in Ollama
```

## Production Deployment

For production deployment:

1. Use secure database credentials
2. Set `ENVIRONMENT=production` in `.env`
3. Configure proper CORS origins in backend
4. Use reverse proxy (nginx) with SSL/TLS
5. Set up monitoring and logging
6. Use managed databases for Neo4j
7. Scale Ollama separately if needed

## License

MIT

## Support

For issues and questions, check the documentation or create an issue.
