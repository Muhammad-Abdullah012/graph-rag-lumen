# Project Structure & File Overview

## Backend (Python/FastAPI)

### Core Modules (`backend/app/modules/`)
- **extraction.py** - PDF text extraction using Docling
  - PDFExtractor class with OCR, tables, and code enrichment
  - Handles German language documents
  - Singleton pattern for resource efficiency

- **splitting.py** - Markdown text splitting with structure preservation
  - MarkdownHeaderTextSplitter integration
  - Recursive splitting for large chunks
  - Metadata preservation

- **database.py** - Neo4j connection and management
  - Connection pooling
  - Constraint creation
  - Vector index management
  - CRUD operations

- **ollama_client.py** - Ollama LLM integration
  - Embedding generation (nomic-embed-text)
  - Text generation (llama3.2 for Q&A)
  - Model listing and verification
  - Ngrok tunnel support

- **graph_builder.py** - Knowledge graph construction
  - LLMGraphTransformer integration
  - Entity and relationship extraction
  - Document chunk node creation
  - Vector embedding storage

- **qa.py** - Question answering system
  - Vector similarity search
  - Keyword search fallback
  - Answer generation with context
  - Source attribution

### API Routes (`backend/app/api/`)
- **documents.py** - Document upload and processing
  - POST /api/documents/upload - File upload with validation
  - GET /api/documents/status/{id} - Processing status
  - Background task processing
  - File size validation

- **qa.py** - Question answering endpoint
  - POST /api/qa/ask - Question submission
  - Source retrieval and attribution
  - Confidence scoring

- **health.py** - Health check endpoint
  - GET /api/health/ - System status
  - Neo4j connection verification
  - Ollama availability check

### Main Application
- **main.py** - FastAPI application entry point
  - CORS middleware
  - Route registration
  - Lifespan management
  - Error handling

### Configuration
- **config/settings.py** - Environment-based configuration
  - Pydantic settings model
  - Neo4j, Ollama, and application settings
  - Type validation

- **config/logging_config.py** - Structured logging setup
  - JSON-formatted logs
  - Log level configuration
  - Third-party logger adjustment

### Docker
- **backend/Dockerfile** - Backend container definition
  - Python 3.11 slim base
  - Dependency installation
  - Health checks

## Frontend (React)

### Components (`frontend/src/components/`)
- **DocumentUpload.jsx** - PDF upload interface
  - File validation and size checking
  - Upload progress
  - Processing status monitoring
  - Supported features list

- **QAChat.jsx** - Q&A chat interface
  - Message history
  - Typing indicator
  - Source attribution display
  - Confidence scoring
  - Auto-scroll to latest message

- **HealthStatus.jsx** - System health monitoring
  - Real-time health checks
  - Service status display
  - Auto-refresh

### Styling (`frontend/src/components/`)
- **DocumentUpload.css** - Upload interface styles
- **QAChat.css** - Chat interface styles
- **HealthStatus.css** - Health indicator styles

### Application
- **App.jsx** - Main React component
  - Tab navigation
  - Component composition
  - State management

- **index.jsx** - React entry point
  - ReactDOM rendering
  - App initialization

- **index.css** - Global styles
  - Layout structure
  - Responsive design
  - Color scheme

### Configuration
- **package.json** - npm dependencies and scripts
- **public/index.html** - HTML template

### Docker
- **frontend/Dockerfile** - Frontend container definition
  - Multi-stage build (node builder + nginx)
  - React build optimization
  - Static file serving

## Infrastructure

### Docker Compose
- **docker-compose.yaml** - Service orchestration
  - neo4j service (enterprise 5.26.1)
  - backend service (FastAPI)
  - frontend service (React)
  - nginx service (reverse proxy)
  - Volume definitions
  - Network configuration
  - Health checks

### Nginx
- **nginx/nginx.conf** - Reverse proxy configuration
  - API routing (/api/ → backend:8000)
  - Frontend routing (/ → frontend:3000)
  - Document serving (/documents/ → local files)
  - CORS headers

## Configuration Files

- **.env.example** - Environment template with all variables
- **.env.neo** - Neo4j-specific configuration
- **.gitignore** - Git ignore patterns

## Scripts

- **scripts/startup.sh** - Automated startup with validation
- **scripts/cleanup.sh** - Service teardown and cleanup
- **scripts/manage_db.py** - Database management utilities
  - Database initialization
  - Statistics reporting
  - Data clearing (with confirmation)

## Documentation

- **README.md** - Complete user and deployment guide
  - Features overview
  - Architecture diagram
  - Quick start instructions
  - Configuration guide
  - API documentation
  - Troubleshooting
  - Production deployment

- **IMPLEMENTATION.md** - Technical implementation details
  - Project overview
  - Technology stack
  - Feature implementation summary
  - Configuration specifications
  - API specifications
  - Data flow diagrams
  - Production readiness checklist
  - Future enhancement opportunities

- **QUICKSTART.sh** - Interactive setup guide
  - Step-by-step initialization
  - Service verification
  - Access point listing
  - Useful commands reference

## File Statistics

- **Python Files**: 13 modules + 1 main
- **React Components**: 3 (DocumentUpload, QAChat, HealthStatus)
- **CSS Files**: 4 (global + 3 component-specific)
- **Configuration Files**: 4 (.env.example, .env.neo, package.json, docker-compose.yaml)
- **Documentation**: 3 files (README, IMPLEMENTATION, QUICKSTART)
- **Utility Scripts**: 3 (startup, cleanup, manage_db)
- **Total Source Files**: 35+

## Key Architectural Patterns

1. **Modular Backend**: Separate modules for each concern (extraction, splitting, DB, LLM, QA)
2. **Singleton Pattern**: Database and Ollama clients to avoid resource exhaustion
3. **Async Processing**: FastAPI + background tasks for long-running operations
4. **Component-Based Frontend**: Reusable React components with local state
5. **Container-Based Deployment**: Docker Compose for easy orchestration
6. **Configuration Management**: Environment-based config with Pydantic validation
7. **Health Checks**: Multi-level health verification (Docker + API endpoint)

## Dependencies Overview

### Backend
- FastAPI, Uvicorn (Web framework)
- Neo4j driver (Database)
- Docling (PDF extraction)
- LangChain (Graph transformation, text splitting)
- Requests (HTTP client)
- Pydantic (Configuration validation)

### Frontend
- React 18 (UI framework)
- React Scripts (Build tools)

### Infrastructure
- Docker, Docker Compose
- Nginx (Reverse proxy)
- Neo4j Enterprise (Graph database)

## Deployment Strategy

1. **Local Development**: Start with `docker-compose up`
2. **Production**: Configure secure credentials, enable SSL/TLS
3. **Scaling**: Neo4j can scale independently, backend is stateless
4. **Monitoring**: Health checks and logging already integrated
5. **Backup**: Configure Docker volume backup strategies

---

This complete project structure is production-ready and designed for scalability and maintainability.
