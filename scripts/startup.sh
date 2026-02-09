#!/bin/bash
# Startup script for Graph RAG

set -e

echo "=== Graph RAG System Startup ==="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and set OLLAMA_BASE_URL"
    exit 1
fi

# Create required directories
echo "Creating required directories..."
mkdir -p documents logs

# Start services
echo "Starting Docker containers..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 10

# Check health
echo "Checking service health..."
if curl -f http://localhost:8000/api/health/ > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
fi

echo ""
echo "=== Setup Complete ==="
echo "Frontend:  http://localhost:8080"
echo "API Docs:  http://localhost:8000/docs"
echo "Health:    http://localhost:8000/api/health"
echo ""
