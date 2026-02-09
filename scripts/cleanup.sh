#!/bin/bash
# Cleanup script

set -e

echo "=== Stopping Graph RAG System ==="

docker-compose down

echo "Removing volumes..."
docker-compose down -v

echo "=== Cleanup Complete ==="
