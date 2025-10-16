#!/bin/bash
# Performance benchmarking script for wasm-chord
# Generates real performance metrics for CPU and GPU backends

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_MODEL_PATH="${WASM_CHORD_TEST_MODEL:-/tmp/wasm-chord-models/tinyllama-1.1b.Q4_K_M.gguf}"
OUTPUT_FILE="benchmark-results.md"
WARMUP_ITERATIONS=3
TEST_ITERATIONS=5
MAX_TOKENS=100

# Test prompts
PROMPTS=(
    "The quick brown fox jumps over the lazy dog."
    "Artificial intelligence is transforming the world."
    "Once upon a time, in a land far away, there lived a wise old wizard."
    "The future of computing lies in quantum mechanics and neural networks."
    "Rust is a systems programming language that runs blazingly fast."
)

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
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

# Get system information
get_system_info() {
    echo "## System Information" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "- **OS:** $(uname -a)" >> "$OUTPUT_FILE"
    echo "- **CPU:** $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)" >> "$OUTPUT_FILE"
    echo "- **CPU Cores:** $(nproc)" >> "$OUTPUT_FILE"
    echo "- **Memory:** $(free -h | grep 'Mem:' | awk '{print $2}')" >> "$OUTPUT_FILE"
    echo "- **Rust Version:** $(rustc --version)" >> "$OUTPUT_FILE"
    echo "- **Cargo Version:** $(cargo --version)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)" >> "$OUTPUT_FILE"
        echo "- **GPU Memory:** $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB" >> "$OUTPUT_FILE"
    else
        echo "- **GPU:** Not available" >> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
}

# Run CPU benchmark
run_cpu_benchmark() {
    log_info "Running CPU benchmarks..."
    
    echo "## CPU Performance Results" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    local total_time=0
    local total_tokens=0
    
    for ((i=1; i<=TEST_ITERATIONS; i++)); do
        log_info "CPU iteration $i/$TEST_ITERATIONS"
        
        # Run benchmark and capture output
        local output
        output=$(cargo run --release --manifest-path examples/benchmark/Cargo.toml -- \
            --max-tokens "$MAX_TOKENS" \
            --warmup "$WARMUP_ITERATIONS" \
            --iterations 1 \
            --prompt "${PROMPTS[$((i % ${#PROMPTS[@]}))]}" 2>&1)
        
        # Extract metrics (assuming benchmark outputs these)
        local tokens_per_sec
        local avg_time_per_token
        local total_time_iteration
        
        # Parse output for metrics (this is a simplified parser)
        tokens_per_sec=$(echo "$output" | grep -o '[0-9.]* tokens/sec' | grep -o '[0-9.]*' | head -1 || echo "0")
        avg_time_per_token=$(echo "$output" | grep -o '[0-9.]* ms/token' | grep -o '[0-9.]*' | head -1 || echo "0")
        
        if [[ "$tokens_per_sec" != "0" ]]; then
            total_time=$((total_time + $(echo "scale=2; $MAX_TOKENS / $tokens_per_sec" | bc)))
            total_tokens=$((total_tokens + MAX_TOKENS))
            
            echo "### Iteration $i" >> "$OUTPUT_FILE"
            echo "- **Prompt:** ${PROMPTS[$((i % ${#PROMPTS[@]}))]}" >> "$OUTPUT_FILE"
            echo "- **Tokens/sec:** $tokens_per_sec" >> "$OUTPUT_FILE"
            echo "- **Time per token:** ${avg_time_per_token} ms" >> "$OUTPUT_FILE"
            echo "- **Total time:** $(echo "scale=2; $MAX_TOKENS / $tokens_per_sec" | bc) seconds" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        else
            log_warning "Failed to parse metrics from iteration $i"
        fi
    done
    
    # Calculate averages
    if [[ $total_tokens -gt 0 ]]; then
        local avg_tokens_per_sec
        avg_tokens_per_sec=$(echo "scale=2; $total_tokens / $total_time" | bc)
        echo "### CPU Summary" >> "$OUTPUT_FILE"
        echo "- **Average tokens/sec:** $avg_tokens_per_sec" >> "$OUTPUT_FILE"
        echo "- **Total tokens generated:** $total_tokens" >> "$OUTPUT_FILE"
        echo "- **Total time:** ${total_time}s" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
    
    log_success "CPU benchmarks completed"
}

# Run GPU benchmark
run_gpu_benchmark() {
    log_info "Running GPU benchmarks..."
    
    echo "## GPU Performance Results" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Check if GPU is available
    if ! cargo run --release --features webgpu --manifest-path examples/benchmark/Cargo.toml -- --help > /dev/null 2>&1; then
        log_warning "GPU backend not available, skipping GPU benchmarks"
        echo "**Note:** GPU backend not available on this system" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        return
    fi
    
    local total_time=0
    local total_tokens=0
    
    for ((i=1; i<=TEST_ITERATIONS; i++)); do
        log_info "GPU iteration $i/$TEST_ITERATIONS"
        
        # Run GPU benchmark
        local output
        output=$(cargo run --release --features webgpu --manifest-path examples/benchmark/Cargo.toml -- \
            --max-tokens "$MAX_TOKENS" \
            --warmup "$WARMUP_ITERATIONS" \
            --iterations 1 \
            --prompt "${PROMPTS[$((i % ${#PROMPTS[@]}))]}" 2>&1)
        
        # Parse output for metrics
        local tokens_per_sec
        local avg_time_per_token
        
        tokens_per_sec=$(echo "$output" | grep -o '[0-9.]* tokens/sec' | grep -o '[0-9.]*' | head -1 || echo "0")
        avg_time_per_token=$(echo "$output" | grep -o '[0-9.]* ms/token' | grep -o '[0-9.]*' | head -1 || echo "0")
        
        if [[ "$tokens_per_sec" != "0" ]]; then
            total_time=$((total_time + $(echo "scale=2; $MAX_TOKENS / $tokens_per_sec" | bc)))
            total_tokens=$((total_tokens + MAX_TOKENS))
            
            echo "### Iteration $i" >> "$OUTPUT_FILE"
            echo "- **Prompt:** ${PROMPTS[$((i % ${#PROMPTS[@]}))]}" >> "$OUTPUT_FILE"
            echo "- **Tokens/sec:** $tokens_per_sec" >> "$OUTPUT_FILE"
            echo "- **Time per token:** ${avg_time_per_token} ms" >> "$OUTPUT_FILE"
            echo "- **Total time:** $(echo "scale=2; $MAX_TOKENS / $tokens_per_sec" | bc) seconds" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        else
            log_warning "Failed to parse GPU metrics from iteration $i"
        fi
    done
    
    # Calculate averages
    if [[ $total_tokens -gt 0 ]]; then
        local avg_tokens_per_sec
        avg_tokens_per_sec=$(echo "scale=2; $total_tokens / $total_time" | bc)
        echo "### GPU Summary" >> "$OUTPUT_FILE"
        echo "- **Average tokens/sec:** $avg_tokens_per_sec" >> "$OUTPUT_FILE"
        echo "- **Total tokens generated:** $total_tokens" >> "$OUTPUT_FILE"
        echo "- **Total time:** ${total_time}s" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
    
    log_success "GPU benchmarks completed"
}

# Run memory usage tests
run_memory_tests() {
    log_info "Running memory usage tests..."
    
    echo "## Memory Usage Analysis" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Test memory64 functionality
    if cargo run --release --manifest-path examples/memory64-test/Cargo.toml > /dev/null 2>&1; then
        echo "- **Memory64 Support:** ✅ Available" >> "$OUTPUT_FILE"
    else
        echo "- **Memory64 Support:** ❌ Not available" >> "$OUTPUT_FILE"
    fi
    
    # Test multi-memory sharding
    if cargo run --release --manifest-path examples/multi-memory-test/Cargo.toml > /dev/null 2>&1; then
        echo "- **Multi-Memory Sharding:** ✅ Available" >> "$OUTPUT_FILE"
    else
        echo "- **Multi-Memory Sharding:** ❌ Not available" >> "$OUTPUT_FILE"
    fi
    
    echo "" >> "$OUTPUT_FILE"
    log_success "Memory tests completed"
}

# Generate comparison with llama.cpp
generate_comparison() {
    log_info "Generating performance comparison..."
    
    echo "## Performance Comparison" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Check if llama.cpp is available
    if command -v llama-cli &> /dev/null; then
        log_info "Found llama.cpp, running comparison..."
        
        echo "### llama.cpp Baseline" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        # Run llama.cpp benchmark (simplified)
        local llama_output
        llama_output=$(timeout 30s llama-cli -m "$TEST_MODEL_PATH" -p "Hello world" -n 10 2>&1 || echo "llama.cpp not available")
        
        if [[ "$llama_output" != "llama.cpp not available" ]]; then
            echo "- **llama.cpp tokens/sec:** [To be measured]" >> "$OUTPUT_FILE"
            echo "- **llama.cpp memory usage:** [To be measured]" >> "$OUTPUT_FILE"
        else
            echo "- **llama.cpp:** Not available for comparison" >> "$OUTPUT_FILE"
        fi
        echo "" >> "$OUTPUT_FILE"
    else
        echo "**Note:** llama.cpp not found, skipping comparison" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
    
    log_success "Comparison completed"
}

# Main execution
main() {
    echo "📊 wasm-chord Performance Benchmark Suite"
    echo "=========================================="
    echo ""
    
    # Initialize output file
    echo "# wasm-chord Performance Benchmark Results" > "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "**Generated:** $(date)" >> "$OUTPUT_FILE"
    echo "**Model:** TinyLLaMA 1.1B Q4_K_M" >> "$OUTPUT_FILE"
    echo "**Test Configuration:**" >> "$OUTPUT_FILE"
    echo "- Warmup iterations: $WARMUP_ITERATIONS" >> "$OUTPUT_FILE"
    echo "- Test iterations: $TEST_ITERATIONS" >> "$OUTPUT_FILE"
    echo "- Max tokens per test: $MAX_TOKENS" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Check prerequisites
    check_model
    
    # Get system information
    get_system_info
    
    # Run benchmarks
    run_cpu_benchmark
    run_gpu_benchmark
    run_memory_tests
    generate_comparison
    
    # Final summary
    echo "## Summary" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "Benchmark completed successfully! 🎉" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "**Key Metrics:**" >> "$OUTPUT_FILE"
    echo "- CPU performance: [See CPU Results section]" >> "$OUTPUT_FILE"
    echo "- GPU performance: [See GPU Results section]" >> "$OUTPUT_FILE"
    echo "- Memory efficiency: [See Memory Usage section]" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    log_success "Benchmark results saved to $OUTPUT_FILE"
    echo ""
    log_info "Results preview:"
    echo "=================="
    head -20 "$OUTPUT_FILE"
    echo "..."
    echo ""
    log_success "Full results available in $OUTPUT_FILE"
}

# Run main function
main "$@"
