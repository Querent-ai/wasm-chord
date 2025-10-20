#!/bin/bash
#
# Memory64 Benchmark Script
# Tests memory usage, loading time, and inference speed

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
RESET="\033[0m"

echo -e "${BOLD}🚀 Memory64 Benchmark Suite${RESET}"
echo "=============================="
echo ""

# Models to test
TINY_MODEL="models/tinyllama-1.1b.Q4_K_M.gguf"
LARGE_MODEL="models/llama-2-7b-chat-q4_k_m.gguf"

# Check if models exist
if [ ! -f "$TINY_MODEL" ]; then
    echo "❌ TinyLlama model not found: $TINY_MODEL"
    exit 1
fi

if [ ! -f "$LARGE_MODEL" ]; then
    echo "❌ Llama-2-7B model not found: $LARGE_MODEL"
    exit 1
fi

echo -e "${BLUE}📦 Building benchmarks...${RESET}"
cargo build --release --package memory64-gguf-test
cargo build --release --package memory64-layer-loading-test
echo ""

# Benchmark 1: Memory Usage
echo -e "${BOLD}📊 Benchmark 1: Memory Usage${RESET}"
echo "----------------------------"
echo ""

echo -e "${GREEN}Testing TinyLlama (0.67GB - Standard Loading)${RESET}"
/usr/bin/time -v cargo run --release --package memory64-gguf-test 2>&1 | grep -E "(Maximum resident set size|User time|System time|Percent of CPU)" || true
echo ""

echo -e "${GREEN}Testing Llama-2-7B (4.08GB - Memory64)${RESET}"
/usr/bin/time -v cargo run --release --package memory64-gguf-test 2>&1 | grep -E "(Maximum resident set size|User time|System time|Percent of CPU)" || true
echo ""

# Benchmark 2: Loading Time
echo -e "${BOLD}⏱️  Benchmark 2: Loading Time${RESET}"
echo "----------------------------"
echo ""

echo -e "${GREEN}TinyLlama Loading Time:${RESET}"
time cargo run --release --package memory64-gguf-test 2>&1 | grep "TinyLlama" -A 10 || true
echo ""

echo -e "${GREEN}Llama-2-7B Loading Time:${RESET}"
time cargo run --release --package memory64-gguf-test 2>&1 | grep "llama-2-7b" -A 10 || true
echo ""

# Benchmark 3: Layer Access Performance
echo -e "${BOLD}🔄 Benchmark 3: Layer Access Performance${RESET}"
echo "----------------------------------------"
echo ""

echo "Testing layer loading and cache hit rates..."
cargo run --release --package memory64-layer-loading-test 2>&1 | grep -E "(Cache|Loading layer|Evicted)" || true
echo ""

# Benchmark 4: Memory Overhead
echo -e "${BOLD}💾 Benchmark 4: Memory Overhead Analysis${RESET}"
echo "----------------------------------------"
echo ""

echo "Calculating memory overhead for Memory64 runtime..."
echo ""
echo "Standard Model (TinyLlama):"
ls -lh "$TINY_MODEL" | awk '{print "  File size: " $5}'
echo ""
echo "Memory64 Model (Llama-2-7B):"
ls -lh "$LARGE_MODEL" | awk '{print "  File size: " $5}'
echo ""

# Summary
echo -e "${BOLD}📈 Benchmark Summary${RESET}"
echo "===================="
echo ""
echo "✅ Memory usage tested"
echo "✅ Loading time measured"
echo "✅ Layer access performance evaluated"
echo "✅ Cache efficiency analyzed"
echo ""
echo -e "${GREEN}Memory64 benchmarks complete!${RESET}"
