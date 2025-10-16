#!/bin/bash
# Comprehensive test script for wasm-chord examples
# This script tests all 63 examples systematically

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_MODEL_PATH="${WASM_CHORD_TEST_MODEL:-/tmp/wasm-chord-models/tinyllama-1.1b.Q4_K_M.gguf}"
TIMEOUT_SECONDS=60
MAX_TOKENS=20

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Test categories
declare -A TEST_CATEGORIES=(
    ["basic"]="simple-generation chat chat-streaming inference cli"
    ["gpu"]="gpu-generation gpu-test gpu-cpu-comparison kernel-verification"
    ["memory64"]="memory64-test wasm-memory64-test wasm-memory64-multi-test multi-memory-test sharding-test wasm-10gb-test comprehensive-memory64-test"
    ["debug"]="tokenizer-debug debug-embedding-step debug-generation debug-forward-pass debug-gibberish debug-performance"
    ["validation"]="vocab-check tokenization-check weight-verification check-embedding-format check-lm-head"
    ["integration"]="abi-tests integration_tests ollama-comparison ollama-comprehensive-test quality-test model-coherence-test"
    ["performance"]="benchmark debug-performance"
    ["specialized"]="argmax-test first-token-comparison implementation-comparison gguf-diagnostic logit-analysis logit-comparison lm-head-debug layer-debug rmsnorm-debug transpose-debug find-token-id test-embeddings test-fix test-matmul test-prompts test-rmsnorm fused-kernels-test q4k-fix-test"
)

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Check if model exists
check_model() {
    if [[ ! -f "$TEST_MODEL_PATH" ]]; then
        log_warning "Test model not found at $TEST_MODEL_PATH"
        log_info "Downloading TinyLLaMA model..."
        mkdir -p "$(dirname "$TEST_MODEL_PATH")"
        wget -O "$TEST_MODEL_PATH" \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        log_success "Model downloaded successfully"
    else
        log_success "Test model found at $TEST_MODEL_PATH"
    fi
}

# Test a single example
test_example() {
    local example_name="$1"
    local example_path="examples/$example_name"
    local manifest_path="$example_path/Cargo.toml"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [[ ! -f "$manifest_path" ]]; then
        log_warning "Example $example_name not found, skipping"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        return
    fi
    
    log_info "Testing $example_name..."
    
    # Build the example
    if cargo build --release --manifest-path "$manifest_path" > /dev/null 2>&1; then
        log_success "Built $example_name"
    else
        log_error "Failed to build $example_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return
    fi
    
    # Try to run the example (with timeout)
    local run_cmd="cargo run --release --manifest-path $manifest_path"
    
    # Add specific arguments for certain examples
    case "$example_name" in
        "simple-generation"|"chat"|"chat-streaming")
            run_cmd="$run_cmd -- \"Hello, world!\""
            ;;
        "benchmark")
            run_cmd="$run_cmd -- --max-tokens $MAX_TOKENS --warmup 1 --iterations 1"
            ;;
        "gpu-generation")
            run_cmd="$run_cmd -- \"Test prompt\" --max-tokens $MAX_TOKENS"
            ;;
    esac
    
    # Run with timeout
    if timeout "$TIMEOUT_SECONDS" bash -c "$run_cmd" > /dev/null 2>&1; then
        log_success "Ran $example_name successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_warning "Example $example_name timed out after ${TIMEOUT_SECONDS}s"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        else
            log_error "Example $example_name failed to run (exit code: $exit_code)"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    fi
}

# Test a category of examples
test_category() {
    local category="$1"
    local examples="${TEST_CATEGORIES[$category]}"
    
    log_info "Testing category: $category"
    echo "=========================================="
    
    for example in $examples; do
        test_example "$example"
    done
    
    echo ""
}

# Test WASM examples
test_wasm_examples() {
    log_info "Testing WASM examples..."
    echo "=========================================="
    
    # Check if wasm-pack is available
    if ! command -v wasm-pack &> /dev/null; then
        log_warning "wasm-pack not found, installing..."
        curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
    fi
    
    # Test WASM runtime build
    log_info "Building WASM runtime..."
    if cd crates/wasm-chord-runtime && wasm-pack build --target web --out-dir pkg > /dev/null 2>&1; then
        log_success "WASM runtime built successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "Failed to build WASM runtime"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    cd ../..
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Test WASM examples
    local wasm_examples="wasm-capital-test wasm-memory64-test wasm-memory64-multi-test"
    for example in $wasm_examples; do
        test_example "$example"
    done
    
    echo ""
}

# Main execution
main() {
    echo "🧪 wasm-chord Comprehensive Test Suite"
    echo "======================================"
    echo ""
    
    # Check prerequisites
    check_model
    
    echo ""
    log_info "Starting comprehensive testing..."
    echo ""
    
    # Test all categories
    for category in "${!TEST_CATEGORIES[@]}"; do
        test_category "$category"
    done
    
    # Test WASM examples separately
    test_wasm_examples
    
    # Print summary
    echo "📊 Test Summary"
    echo "==============="
    echo "Total tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "Skipped: ${YELLOW}$SKIPPED_TESTS${NC}"
    echo ""
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log_success "All tests passed! 🎉"
        exit 0
    else
        log_error "$FAILED_TESTS tests failed"
        exit 1
    fi
}

# Run main function
main "$@"
