#!/bin/bash
#
# Compare Memory64 performance with and without async prefetching
#

set -e

echo "🔬 Memory64 Async Prefetch Performance Comparison"
echo "=================================================="
echo ""

MODEL="${1:-models/llama-2-7b-chat-q4_k_m.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo "Usage: $0 [MODEL_PATH]"
    exit 1
fi

echo "📂 Model: $MODEL"
echo ""

# Build both versions
echo "🔨 Building test binaries..."
echo ""

echo "  Building WITHOUT async-prefetch..."
cd examples/memory64-model-test
cargo build --release --quiet
SYNC_BIN="../../target/release/memory64-model-test"

echo "  Building WITH async-prefetch..."
cargo build --release --features async-prefetch --quiet
ASYNC_BIN="../../target/release/memory64-model-test"

cd ../..

echo "✅ Build complete"
echo ""

# Run synchronous test
echo "📊 Test 1: Synchronous Prefetch (baseline)"
echo "==========================================="
echo ""

SYNC_LOG=$(mktemp)
timeout 45 $SYNC_BIN "$MODEL" 2>&1 | tee $SYNC_LOG || true

SYNC_LOADS=$(grep -c "Loading layer.*sync" $SYNC_LOG || echo "0")
SYNC_TIME=$(grep "Time:" $SYNC_LOG | awk '{print $3}' || echo "N/A")

echo ""
echo "  ⏱️  Time: $SYNC_TIME"
echo "  📦 Synchronous loads: $SYNC_LOADS"
echo ""

# Run async test
echo "📊 Test 2: Async Background Prefetch"
echo "====================================="
echo ""

ASYNC_LOG=$(mktemp)
# Rebuild with async feature
cd examples/memory64-model-test
cargo build --release --features async-prefetch --quiet
cd ../..

timeout 45 $ASYNC_BIN "$MODEL" 2>&1 | tee $ASYNC_LOG || true

ASYNC_LOADS=$(grep -c "Loading layer.*sync" $ASYNC_LOG || echo "0")
ASYNC_PREFETCH=$(grep -c "Prefetched layer.*ready" $ASYNC_LOG || echo "0")
ASYNC_TIME=$(grep "Time:" $ASYNC_LOG | awk '{print $3}' || echo "N/A")

echo ""
echo "  ⏱️  Time: $ASYNC_TIME"
echo "  📦 Synchronous loads: $ASYNC_LOADS"
echo "  ⚡ Async prefetch hits: $ASYNC_PREFETCH"
echo ""

# Summary
echo "📈 Summary"
echo "=========="
echo ""
echo "| Metric | Synchronous | Async | Improvement |"
echo "|--------|-------------|-------|-------------|"
echo "| Sync loads | $SYNC_LOADS | $ASYNC_LOADS | $(echo "scale=1; 100 * (1 - $ASYNC_LOADS / $SYNC_LOADS)" | bc)% fewer |"
echo "| Prefetch hits | 0 | $ASYNC_PREFETCH | +$ASYNC_PREFETCH |"
echo ""

if [ "$SYNC_TIME" != "N/A" ] && [ "$ASYNC_TIME" != "N/A" ]; then
    SPEEDUP=$(echo "scale=2; ${SYNC_TIME%s} / ${ASYNC_TIME%s}" | bc)
    echo "⚡ Speedup: ${SPEEDUP}x faster with async prefetch"
else
    echo "⚠️  Could not calculate speedup (tests may have timed out)"
fi

echo ""
echo "✅ Comparison complete!"
echo ""
echo "🧹 Logs saved to:"
echo "  Sync:  $SYNC_LOG"
echo "  Async: $ASYNC_LOG"


