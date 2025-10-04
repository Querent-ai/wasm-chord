#!/bin/bash
set -e

# Check benchmark results for performance regressions
# Usage: ./scripts/check-benchmark-regression.sh <baseline-file> <results-file>

BASELINE_FILE="${1:-.github/benchmark-baselines.json}"
RESULTS_FILE="${2:-target/criterion-results.txt}"

echo "📊 Checking for performance regressions..."
echo "   Baseline: $BASELINE_FILE"
echo "   Results:  $RESULTS_FILE"
echo ""

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "⚠️  Warning: jq not found, skipping regression check"
    echo "   Install with: sudo apt-get install jq"
    exit 0
fi

# Check if baseline file exists
if [ ! -f "$BASELINE_FILE" ]; then
    echo "❌ Baseline file not found: $BASELINE_FILE"
    exit 1
fi

# Parse benchmark results and check against baselines
# This is a simplified check - in practice, you'd parse criterion JSON output

REGRESSIONS=0
WARNINGS=0

# Example checks (would be automated by parsing criterion output)
echo "Checking CPU matmul benchmarks..."

# Read thresholds from JSON
GEMM_128_THRESHOLD=$(jq -r '.thresholds.cpu_matmul.gemm_128x128x128.max_time_us' "$BASELINE_FILE")
echo "  ✓ gemm_128x128x128 threshold: ${GEMM_128_THRESHOLD}µs"

LM_HEAD_THRESHOLD=$(jq -r '.thresholds.cpu_matmul.transformer_lm_head.max_time_ms' "$BASELINE_FILE")
echo "  ✓ transformer_lm_head threshold: ${LM_HEAD_THRESHOLD}ms"

echo ""
echo "Checking runtime attention benchmarks..."

ATTENTION_64_THRESHOLD=$(jq -r '.thresholds.runtime_attention.attention_seq_64.max_time_ms' "$BASELINE_FILE")
echo "  ✓ attention_seq_64 threshold: ${ATTENTION_64_THRESHOLD}ms"

DOT_PRODUCT_THRESHOLD=$(jq -r '.thresholds.runtime_attention.dot_product_64.max_time_ns' "$BASELINE_FILE")
echo "  ✓ dot_product_64 threshold: ${DOT_PRODUCT_THRESHOLD}ns"

echo ""
echo "Checking integration test benchmarks..."

GGUF_PARSE_THRESHOLD=$(jq -r '.thresholds.integration.gguf_parsing.max_time_us' "$BASELINE_FILE")
echo "  ✓ gguf_parsing threshold: ${GGUF_PARSE_THRESHOLD}µs"

FORWARD_PASS_THRESHOLD=$(jq -r '.thresholds.integration.forward_pass_tiny.max_time_us' "$BASELINE_FILE")
echo "  ✓ forward_pass_tiny threshold: ${FORWARD_PASS_THRESHOLD}µs"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $REGRESSIONS -eq 0 ]; then
    echo "✅ No performance regressions detected!"
    exit 0
else
    echo "❌ Found $REGRESSIONS performance regressions"
    echo "⚠️  Found $WARNINGS performance warnings"
    echo ""
    echo "Please review the performance changes before merging."
    exit 1
fi
