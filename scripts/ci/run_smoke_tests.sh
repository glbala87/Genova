#!/bin/bash
# ============================================================================
# Smoke tests for deployed Genova service
# ============================================================================
# Usage: bash scripts/ci/run_smoke_tests.sh <service_url>
#
# Verifies core API endpoints are functional after deployment.
# Exits 0 on success, 1 on any failure.

set -euo pipefail

# ── Color helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${CYAN}==>${NC} $1"; }

# ── Arguments ────────────────────────────────────────────────────────────────
SERVICE_URL="${1:-}"
TIMEOUT="${SMOKE_TEST_TIMEOUT:-10}"

if [ -z "$SERVICE_URL" ]; then
    echo -e "${RED}Error:${NC} Missing required argument <service_url>"
    echo "Usage: bash $0 <service_url>"
    exit 1
fi

# Strip trailing slash
SERVICE_URL="${SERVICE_URL%/}"

FAILURES=0
TOTAL=0

# ── Helper ───────────────────────────────────────────────────────────────────
run_test() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@"; then
        pass "$name"
    else
        fail "$name"
    fi
}

# ── Test: /health ────────────────────────────────────────────────────────────
test_health() {
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT" \
        "${SERVICE_URL}/health")
    [ "$status" = "200" ]
}

# ── Test: /model/info ────────────────────────────────────────────────────────
test_model_info() {
    local response
    response=$(curl -sf --max-time "$TIMEOUT" "${SERVICE_URL}/model/info")
    [ $? -eq 0 ] || return 1

    # Validate it is valid JSON with expected fields
    echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
required = ['model_name', 'version']
# Accept either model_name or name
if 'name' in data and 'model_name' not in data:
    data['model_name'] = data['name']
for field in required:
    if field not in data:
        print(f'Missing field: {field}', file=sys.stderr)
        sys.exit(1)
print('Valid JSON with required fields')
"
}

# ── Test: /predict_variant ───────────────────────────────────────────────────
test_predict_variant() {
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT" \
        -X POST "${SERVICE_URL}/predict_variant" \
        -H "Content-Type: application/json" \
        -d '{
            "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCG",
            "variant_position": 16,
            "ref_allele": "A",
            "alt_allele": "T"
        }')
    [ "$status" = "200" ]
}

# ── Test: /predict_expression ────────────────────────────────────────────────
test_predict_expression() {
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT" \
        -X POST "${SERVICE_URL}/predict_expression" \
        -H "Content-Type: application/json" \
        -d '{
            "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCG"
        }')
    [ "$status" = "200" ]
}

# ── Test: /embed ─────────────────────────────────────────────────────────────
test_embed() {
    local response
    response=$(curl -sf --max-time "$TIMEOUT" \
        -X POST "${SERVICE_URL}/embed" \
        -H "Content-Type: application/json" \
        -d '{
            "sequence": "ATCGATCGATCGATCG"
        }')
    [ $? -eq 0 ] || return 1

    # Verify response contains an embedding array
    echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'embedding' not in data and 'embeddings' not in data:
    print('Response missing embedding field', file=sys.stderr)
    sys.exit(1)
key = 'embedding' if 'embedding' in data else 'embeddings'
emb = data[key]
if not isinstance(emb, list):
    print('Embedding is not a list', file=sys.stderr)
    sys.exit(1)
if len(emb) == 0:
    print('Embedding is empty', file=sys.stderr)
    sys.exit(1)
print(f'Embedding dimension: {len(emb)}')
"
}

# ── Execute tests ────────────────────────────────────────────────────────────
echo ""
info "Running smoke tests against: ${SERVICE_URL}"
echo ""

info "Endpoint tests"
run_test "/health returns 200"               test_health
run_test "/model/info returns valid JSON"    test_model_info
run_test "/predict_variant accepts request"  test_predict_variant
run_test "/predict_expression accepts request" test_predict_expression
run_test "/embed returns embedding vector"   test_embed

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
PASSED=$((TOTAL - FAILURES))
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All ${TOTAL} smoke tests passed.${NC}"
    exit 0
else
    echo -e "${RED}${FAILURES}/${TOTAL} smoke tests failed.${NC}"
    exit 1
fi
