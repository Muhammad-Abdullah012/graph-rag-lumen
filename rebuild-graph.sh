#!/bin/bash
# Rebuild the Neo4j knowledge graph from scratch.
# Connects directly to Neo4j to wipe the graph, then calls the backend API to rebuild it.

set -e

# ── Config (reads from .env if present) ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

NEO4J_CONTAINER="${NEO4J_CONTAINER:-graphrag-neo4j}"
NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jpassword}"
BACKEND_URL="${BACKEND_URL:-http://localhost:9000}"

SKIP_EMBEDDINGS=false
if [[ "$1" == "--no-embeddings" ]]; then
    SKIP_EMBEDDINGS=true
fi

echo "=== Graph Rebuild ==="
echo ""

# ── Step 1: Verify Neo4j container is running ──────────────────────────────────
echo "Step 1: Checking Neo4j container..."
if ! docker ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
    echo "  ERROR: Container '${NEO4J_CONTAINER}' is not running."
    echo "         Start services first: docker compose up -d"
    exit 1
fi
echo "  OK: ${NEO4J_CONTAINER} is running"

# ── Step 2: Wait for Neo4j to accept connections ───────────────────────────────
echo ""
echo "Step 2: Waiting for Neo4j to be ready..."
MAX_WAIT=60
WAITED=0
until docker exec "$NEO4J_CONTAINER" \
    cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
    "RETURN 1" > /dev/null 2>&1; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "  ERROR: Neo4j did not become ready within ${MAX_WAIT}s."
        exit 1
    fi
    echo -n "."
    sleep 2
    WAITED=$((WAITED + 2))
done
echo ""
echo "  OK: Neo4j is ready"

# ── Step 3: Delete the graph ───────────────────────────────────────────────────
echo ""
echo "Step 3: Deleting all nodes and relationships..."
NODE_COUNT=$(docker exec "$NEO4J_CONTAINER" \
    cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
    "MATCH (n) RETURN count(n) AS c" --format plain 2>/dev/null | tail -1 || echo "0")
echo "  Found ${NODE_COUNT} nodes — deleting..."

docker exec "$NEO4J_CONTAINER" \
    cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
    "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"

echo "  OK: Graph cleared"

# ── Step 4: Verify backend is reachable ────────────────────────────────────────
echo ""
echo "Step 4: Checking backend API..."
if ! curl -sf "${BACKEND_URL}/api/health/" > /dev/null 2>&1; then
    echo "  ERROR: Backend not reachable at ${BACKEND_URL}"
    echo "         Is the backend container running?"
    exit 1
fi
echo "  OK: Backend is reachable"

# ── Step 5: Rebuild the graph ──────────────────────────────────────────────────
echo ""
echo "Step 5: Rebuilding graph from JSON files..."
REBUILD_RESPONSE=$(curl -sf -X POST "${BACKEND_URL}/api/graph/build" \
    -H "Content-Type: application/json" 2>&1)

if [ $? -ne 0 ]; then
    echo "  ERROR: Graph build request failed."
    echo "         Response: ${REBUILD_RESPONSE}"
    exit 1
fi

STATUS=$(echo "$REBUILD_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")
if [[ "$STATUS" != "success" ]]; then
    echo "  ERROR: Graph build returned status: ${STATUS}"
    echo "         Full response: ${REBUILD_RESPONSE}"
    exit 1
fi

echo "  OK: Graph built"
echo "  Stats: $(echo "$REBUILD_RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin).get('stats', {})
print(', '.join(f'{k}={v}' for k, v in d.items()))
" 2>/dev/null)"

# ── Step 6: Generate embeddings (optional) ────────────────────────────────────
if [ "$SKIP_EMBEDDINGS" = false ]; then
    echo ""
    echo "Step 6: Generating embeddings..."
    echo "  (pass --no-embeddings to skip this step)"
    EMB_RESPONSE=$(curl -sf -X POST "${BACKEND_URL}/api/graph/generate-embeddings" \
        -H "Content-Type: application/json" 2>&1)
    if [ $? -eq 0 ]; then
        EMB_COUNT=$(echo "$EMB_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('embeddings_created',0))" 2>/dev/null || echo "?")
        echo "  OK: ${EMB_COUNT} embeddings created"
    else
        echo "  WARNING: Embedding generation failed (non-fatal)"
        echo "           Run manually: curl -X POST ${BACKEND_URL}/api/graph/generate-embeddings"
    fi
else
    echo ""
    echo "Step 6: Skipping embeddings (--no-embeddings)"
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== Rebuild Complete ==="
echo ""
echo "Useful commands:"
echo "  Graph stats:  curl ${BACKEND_URL}/api/graph/stats"
echo "  Embeddings:   curl -X POST ${BACKEND_URL}/api/graph/generate-embeddings"
echo "  Similarity:   curl -X POST ${BACKEND_URL}/api/graph/compute-similarity"
echo ""
