#!/bin/bash
# 🚀 GRAPH RAG - GETTING STARTED GUIDE
# 
# This file contains all the exact commands you need to run
# to set up and launch your Graph RAG system.
#
# Copy and paste commands as needed!

echo "=========================================="
echo "Graph RAG System - Getting Started"
echo "=========================================="
echo ""

# ============================================
# STEP 1: INITIAL SETUP
# ============================================
echo "STEP 1: INITIAL SETUP"
echo "===================="
echo ""
echo "Run these commands in the project root:"
echo ""
echo "  # Copy environment template"
echo "  cp .env.example .env"
echo ""
echo "  # Edit .env with your Ollama ngrok URL"
echo "  nano .env"
echo "    # Find: OLLAMA_BASE_URL=https://26ae-2407-d000-1a-16b1-add8-a30a-4f8b-7ef0.ngrok-free.app/"
echo "    # Change to your actual ngrok URL"
echo ""
echo "  # Create required directories"
echo "  mkdir -p documents logs"
echo ""

# ============================================
# STEP 2: VERIFY OLLAMA
# ============================================
echo "STEP 2: VERIFY OLLAMA"
echo "===================="
echo ""
echo "Make sure Ollama is running with required models:"
echo ""
echo "  # Test Ollama connection (replace URL with yours)"
echo "  curl https://your-ngrok-url/api/tags"
echo ""
echo "  # Expected output should include:"
echo "  #   - llama3.4"
echo "  #   - llama3.2"
echo "  #   - nomic-embed-text:latest"
echo ""

# ============================================
# STEP 3: BUILD & START
# ============================================
echo "STEP 3: BUILD & START SYSTEM"
echo "============================"
echo ""
echo "Option A - Automated (Recommended):"
echo "  chmod +x QUICKSTART.sh"
echo "  ./QUICKSTART.sh"
echo ""
echo "Option B - Manual:"
echo "  docker-compose build"
echo "  docker-compose up -d"
echo ""

# ============================================
# STEP 4: VERIFY SERVICES
# ============================================
echo "STEP 4: VERIFY SERVICES"
echo "======================="
echo ""
echo "Check all containers are running:"
echo "  docker-compose ps"
echo ""
echo "Wait ~30 seconds for services to fully start, then test:"
echo ""
echo "  # Test Backend"
echo "  curl http://localhost:8000/api/health/"
echo ""
echo "  # View API Documentation"
echo "  open http://localhost:8000/docs"
echo ""
echo "  # Test Frontend"
echo "  open http://localhost:8080"
echo ""

# ============================================
# STEP 5: UPLOAD & TEST
# ============================================
echo "STEP 5: UPLOAD & TEST DOCUMENTS"
echo "================================"
echo ""
echo "Via Web UI:"
echo "  1. Open http://localhost:8080 in browser"
echo "  2. Click 'Upload Documents' tab"
echo "  3. Select a PDF from: ./input/"
echo "  4. Click 'Upload & Process'"
echo "  5. Wait for processing to complete"
echo ""
echo "Via API (curl):"
echo "  curl -X POST -F 'file=@input/your-document.pdf' \\"
echo "    http://localhost:8000/api/documents/upload"
echo ""

# ============================================
# STEP 6: ASK QUESTIONS
# ============================================
echo "STEP 6: ASK QUESTIONS"
echo "===================="
echo ""
echo "Via Web UI:"
echo "  1. Click 'Ask Questions' tab"
echo "  2. Type a question about your documents"
echo "  3. Click Send (🚀 button)"
echo "  4. View answer with source documents"
echo ""
echo "Via API (curl):"
echo "  curl -X POST -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\": \"What is the main topic?\", \"top_k\": 5}' \\"
echo "    http://localhost:8000/api/qa/ask"
echo ""

# ============================================
# USEFUL COMMANDS
# ============================================
echo "USEFUL COMMANDS"
echo "==============="
echo ""
echo "View logs:"
echo "  docker-compose logs -f backend      # Backend logs"
echo "  docker-compose logs -f neo4j        # Neo4j logs"
echo "  docker-compose logs -f frontend     # Frontend logs"
echo ""
echo "Stop system:"
echo "  docker-compose down"
echo ""
echo "Clean everything (CAUTION - removes data):"
echo "  docker-compose down -v"
echo ""
echo "Check database:"
echo "  open http://localhost:8474  # Neo4j browser"
echo "  Login: neo4j / neo4jpassword"
echo ""
echo "Database utilities:"
echo "  python scripts/manage_db.py init   # Initialize DB"
echo "  python scripts/manage_db.py stats  # Show statistics"
echo "  python scripts/manage_db.py clear  # Clear all data"
echo ""

# ============================================
# TROUBLESHOOTING
# ============================================
echo "TROUBLESHOOTING"
echo "==============="
echo ""
echo "Q: Ollama connection refused"
echo "A: Check ngrok tunnel is active and URL is correct in .env"
echo ""
echo "Q: Port 8000/3000/8080 already in use"
echo "A: Stop other services or change ports in docker-compose.yaml"
echo ""
echo "Q: Neo4j won't start"
echo "A: Check /logs directory exists, try: docker-compose down -v && docker-compose up"
echo ""
echo "Q: PDF processing fails"
echo "A: Check backend logs: docker-compose logs -f backend"
echo ""
echo "Q: Q&A returns no results"
echo "A: Ensure document is fully processed (check status)"
echo ""

# ============================================
# SAMPLE COMMANDS
# ============================================
echo "SAMPLE COMMANDS - COPY & PASTE"
echo "=============================="
echo ""
echo "Complete setup from scratch:"
echo "  cd /home/office/Projects/graph-rag"
echo "  cp .env.example .env"
echo "  # Edit .env with your Ollama URL"
echo "  mkdir -p documents logs"
echo "  chmod +x QUICKSTART.sh"
echo "  ./QUICKSTART.sh"
echo ""
echo "Quick test of all endpoints:"
echo "  # Health check"
echo "  curl http://localhost:8000/api/health/ | python -m json.tool"
echo ""
echo "  # Upload document"
echo "  curl -X POST -F 'file=@input/BEM-ING-Anlage_1_zum_ARS_22_2012-Entwurf_2.pdf' \\"
echo "    http://localhost:8000/api/documents/upload | python -m json.tool"
echo ""
echo "  # Check status (replace DOCUMENT_ID with returned ID)"
echo "  curl http://localhost:8000/api/documents/status/DOCUMENT_ID | python -m json.tool"
echo ""
echo "  # Ask question"
echo "  curl -X POST -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\": \"Was ist das Thema?\", \"top_k\": 5}' \\"
echo "    http://localhost:8000/api/qa/ask | python -m json.tool"
echo ""

# ============================================
# DOCUMENTATION
# ============================================
echo "DOCUMENTATION"
echo "============="
echo ""
echo "📚 Available Documentation:"
echo "  - README.md              Full guide and deployment info"
echo "  - IMPLEMENTATION.md      Technical implementation details"
echo "  - PROJECT_STRUCTURE.md   File organization"
echo "  - COMPLETION_SUMMARY.md  What was built"
echo "  - VERIFICATION.md        Launch checklist"
echo "  - STATUS.md              Current status"
echo ""
echo "🌐 Web Documentation:"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Neo4j: http://localhost:8474 (admin: neo4j)"
echo ""

# ============================================
# FINAL NOTE
# ============================================
echo "=========================================="
echo "Ready to launch! Follow the steps above.  "
echo ""
echo "Questions? Check the documentation files."
echo "=========================================="
echo ""
