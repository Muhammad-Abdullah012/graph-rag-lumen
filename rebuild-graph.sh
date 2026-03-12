#!/bin/bash
# Rebuild the Neo4j knowledge graph from scratch.
# Calls POST /api/graph/rebuild which runs the full pipeline:
#   clear → build nodes → summaries → embeddings → semantic similarity
# Then polls /api/graph/job-status until the job completes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

BACKEND_URL="${BACKEND_URL:-http://localhost:9000}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"  # seconds between status checks

echo "=== Graph Rebuild Pipeline ==="
echo "  Backend: ${BACKEND_URL}"
echo ""

# ── Step 1: Verify backend is reachable ────────────────────────────────────────
echo "Step 1: Checking backend API..."
if ! curl -sf "${BACKEND_URL}/api/health/" > /dev/null 2>&1; then
    echo "  ERROR: Backend not reachable at ${BACKEND_URL}"
    echo "         Is the backend container running?"
    exit 1
fi
echo "  OK: Backend is reachable"

# ── Step 2: Trigger rebuild (returns 202 immediately) ─────────────────────────
echo ""
echo "Step 2: Triggering full rebuild pipeline..."
RESPONSE=$(curl -sf -X POST "${BACKEND_URL}/api/graph/rebuild" \
    -H "Content-Type: application/json" 2>&1)

if [ $? -ne 0 ]; then
    echo "  ERROR: Rebuild request failed."
    echo "         Response: ${RESPONSE}"
    exit 1
fi

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id','?'))" 2>/dev/null || echo "?")
echo "  OK: Job started (id=${JOB_ID})"
echo "  Pipeline: clear → build nodes → summaries → embeddings → similarity"

# ── Step 3: Poll until job finishes ───────────────────────────────────────────
echo ""
echo "Step 3: Waiting for pipeline to complete (polling every ${POLL_INTERVAL}s)..."
echo "        Monitor logs: docker compose logs -f backend"
echo ""

ELAPSED=0
while true; do
    STATUS_JSON=$(curl -sf "${BACKEND_URL}/api/graph/job-status" 2>/dev/null || echo "{}")
    STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")

    if [ "$STATUS" = "success" ]; then
        echo ""
        echo "  OK: Pipeline completed successfully after ${ELAPSED}s"
        echo "  Result: $(echo "$STATUS_JSON" | python3 -c "
import sys, json
r = json.load(sys.stdin).get('result', {})
print(', '.join(f'{k}={v}' for k, v in r.items()))
" 2>/dev/null)"
        break
    elif [ "$STATUS" = "failed" ]; then
        ERROR=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown error'))" 2>/dev/null || echo "unknown error")
        echo ""
        echo "  ERROR: Pipeline failed after ${ELAPSED}s"
        echo "         Error: ${ERROR}"
        exit 1
    else
        printf "  [%ds] Running... (status=%s)\r" "$ELAPSED" "$STATUS"
        sleep "$POLL_INTERVAL"
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
    fi
done

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== Rebuild Complete ==="
echo ""
echo "Useful commands:"
echo "  Graph stats:   curl ${BACKEND_URL}/api/graph/stats"
echo "  Job history:   curl ${BACKEND_URL}/api/graph/job-history"
echo "  Job status:    curl ${BACKEND_URL}/api/graph/job-status"
echo ""
