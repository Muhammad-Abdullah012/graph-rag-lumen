#!/bin/bash
# Quick Start Guide - Execute these commands

set -e

echo "=== Graph RAG Quick Start ==="
echo ""

# Step 1: Environment Setup
echo "Step 1: Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file"
    echo "  ⚠️  IMPORTANT: Edit .env and update OLLAMA_BASE_URL"
    echo "     Format: https://your-ngrok-url/"
    read -p "Press enter after updating .env..."
else
    echo "✓ .env file already exists"
fi

# Step 2: Create directories
echo ""
echo "Step 2: Creating required directories..."
mkdir -p documents logs
echo "✓ Created documents/ and logs/"

# Step 3: Build images
echo ""
echo "Step 3: Building Docker images..."
echo "This may take a few minutes..."
docker compose build
echo "✓ Docker images built"

# Step 4: Start services
echo ""
echo "Step 4: Starting services..."
docker compose up -d
echo "✓ Services started"

# Step 5: Wait for startup
echo ""
echo "Step 5: Waiting for services to initialize..."
echo "Neo4j takes ~30 seconds to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health/ > /dev/null 2>&1; then
        echo "✓ Backend is ready"
        break
    fi
    echo -n "."
    sleep 1
done

# Step 6: Show status
echo ""
echo "Step 6: Checking service status..."
docker compose ps

# Step 7: Health check
echo ""
echo "Step 7: Health check..."
if curl -f http://localhost:8000/api/health/ > /dev/null 2>&1; then
    echo "✓ All services are healthy"
else
    echo "⚠️  Some services may still be starting..."
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "📍 Access Points:"
echo "   Frontend:  http://localhost:8080"
echo "   API Docs:  http://localhost:8000/docs"
echo "   Health:    http://localhost:8000/api/health"
echo ""
echo "📝 Next Steps:"
echo "   1. Open http://localhost:8080 in your browser"
echo "   2. Go to 'Upload Documents' tab"
echo "   3. Upload a PDF (samples in input/ folder)"
echo "   4. Wait for processing to complete"
echo "   5. Ask questions in 'Ask Questions' tab"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs:     docker-compose logs -f backend"
echo "   Stop system:   docker-compose down"
echo "   Clean data:    docker-compose down -v"
echo ""
echo "📚 Documentation:"
echo "   - README.md: Full documentation"
echo "   - IMPLEMENTATION.md: Technical details"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
