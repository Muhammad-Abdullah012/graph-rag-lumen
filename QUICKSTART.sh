#!/bin/bash
# Quick Start - Graph RAG Lumen

set -e

echo "=== Graph RAG Lumen - Quick Start ==="
echo ""

# Step 1: Environment Setup
echo "Step 1: Setting up environment..."
NEED_INPUT=false

if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example"
    NEED_INPUT=true
else
    echo "✓ .env already exists"
fi

if [ ! -f .env.neo ]; then
    cp .env.neo.example .env.neo
    echo "✓ Created .env.neo from .env.neo.example"
    NEED_INPUT=true
else
    echo "✓ .env.neo already exists"
fi

if [ "$NEED_INPUT" = true ]; then
    echo ""
    echo "  ⚠️  IMPORTANT: Edit the config files before continuing:"
    echo "     .env     → MISTRAL_API_KEY, OLLAMA_BASE_URL"
    echo "     .env.neo → NEO4J_AUTH (username/password)"
    echo ""
    read -p "Press Enter after updating the config files..."
fi

# Step 2: Create required directories
echo ""
echo "Step 2: Creating required directories..."
mkdir -p documents logs backend/json backend/images
echo "✓ Created documents/, logs/, backend/json/, backend/images/"

# Step 3: Build images
echo ""
echo "Step 3: Building Docker images (this may take a few minutes)..."
docker compose build
echo "✓ Docker images built"

# Step 4: Start services
echo ""
echo "Step 4: Starting services..."
docker compose up -d
echo "✓ Services started"

# Step 5: Wait for backend
echo ""
echo "Step 5: Waiting for backend to be ready..."
echo "Neo4j takes ~30 seconds to start..."
for i in {1..60}; do
    if curl -sf http://localhost:9000/api/health/ > /dev/null 2>&1; then
        echo ""
        echo "✓ Backend is ready"
        break
    fi
    echo -n "."
    sleep 1
done

# Step 6: Show status
echo ""
echo "Step 6: Service status..."
docker compose ps

# Step 7: Health check
echo ""
echo "Step 7: Health check..."
if curl -sf http://localhost:9000/api/health/ > /dev/null 2>&1; then
    echo "✓ All services healthy"
else
    echo "⚠️  Some services may still be starting. Check: docker compose logs -f backend"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "📍 Access Points:"
echo "   Frontend:    http://localhost:8089"
echo "   API Docs:    http://localhost:9000/docs"
echo "   Health:      http://localhost:9000/api/health/"
echo "   Neo4j:       http://localhost:8475  (bolt: localhost:8688)"
echo ""
echo "📝 Usage:"
echo "   1. Open http://localhost:8089 in your browser"
echo "   2. Go to 'Dateien' tab and upload PDF files"
echo "   3. Click 'Process All Documents' to run OCR and build graph"
echo "   4. Switch to 'Chat' tab and ask questions"
echo ""
echo "🔧 Useful Commands:"
echo "   Logs:           docker compose logs -f backend"
echo "   Stop:           docker compose down"
echo "   Reset graph:    docker compose down -v"
echo "   Rebuild:        docker compose up -d --build"
echo ""
