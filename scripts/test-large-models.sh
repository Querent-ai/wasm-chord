#!/bin/bash
# Large model testing script for wasm-chord
# Tests production readiness with 7B+ models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODELS_DIR="/tmp/wasm-chord-large-models"
TEST_RESULTS_DIR="./large-model-test-results"
TIMEOUT_SECONDS=300
MAX_TOKENS=50

# Model configurations
declare -A LARGE_MODELS=(
    ["llama-2-7b-chat"]="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    ["codellama-7b"]="https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_K_M.gguf"
    ["mistral-7b"]="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)

# Test prompts for different model types
declare -A TEST_PROMPTS=(
    ["general"]="Explain the concept of artificial intelligence in simple terms."
    ["code"]="Write a Python function to calculate the factorial of a number."
    ["reasoning"]="If a train leaves station A at 60 mph and another leaves station B at 40 mph, and they are 200 miles apart, when will they meet?"
    ["creative"]="Write a short story about a robot who discovers emotions."
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

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check available memory
    local total_memory=$(free -g | grep '^Mem:' | awk '{print $2}')
    local available_memory=$(free -g | grep '^Mem:' | awk '{print $7}')
    
    log_info "Total memory: ${total_memory}GB"
    log_info "Available memory: ${available_memory}GB"
    
    if [[ $available_memory -lt 8 ]]; then
        log_warning "Low available memory (${available_memory}GB). Large models may not work properly."
    fi
    
    # Check disk space
    local available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    log_info "Available disk space: ${available_space}GB"
    
    if [[ $available_space -lt 20 ]]; then
        log_warning "Low disk space (${available_space}GB). May not be able to download large models."
    fi
    
    # Check CPU cores
    local cpu_cores=$(nproc)
    log_info "CPU cores: $cpu_cores"
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        log_info "GPU memory: ${gpu_memory}MB"
    else
        log_warning "No NVIDIA GPU detected"
    fi
}

# Download large model
download_model() {
    local model_name="$1"
    local model_url="$2"
    local model_path="$MODELS_DIR/$model_name.gguf"
    
    log_info "Downloading $model_name..."
    
    if [[ -f "$model_path" ]]; then
        log_success "$model_name already exists"
        return 0
    fi
    
    # Create models directory
    mkdir -p "$MODELS_DIR"
    
    # Download with progress
    log_info "Downloading from: $model_url"
    if wget --progress=bar:force -O "$model_path" "$model_url"; then
        log_success "$model_name downloaded successfully"
        
        # Check file size
        local file_size=$(du -h "$model_path" | cut -f1)
        log_info "Model size: $file_size"
        
        return 0
    else
        log_error "Failed to download $model_name"
        return 1
    fi
}

# Test model loading
test_model_loading() {
    local model_name="$1"
    local model_path="$MODELS_DIR/$model_name.gguf"
    local test_name="model-loading-$model_name"
    
    log_info "Testing model loading for $model_name..."
    
    # Create test script
    local test_script="/tmp/test-loading-$model_name.rs"
    cat > "$test_script" << EOF
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig, GenerationConfig};
use std::io::Cursor;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "$model_path";
    let start_time = Instant::now();
    
    // Load model file
    let model_data = std::fs::read(model_path)?;
    let cursor = Cursor::new(model_data);
    let mut parser = GGUFParser::new(cursor);
    
    // Parse header
    let meta = parser.parse_header()?;
    println!("✅ Model header parsed successfully");
    
    // Extract config
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    println!("✅ Model config extracted: {} layers, {} hidden size", config.num_layers, config.hidden_size);
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded successfully");
    
    // Create model
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);
    
    // Register tensors
    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }
    
    // Re-parse for loading
    let cursor = Cursor::new(model_data);
    let mut parser = GGUFParser::new(cursor);
    parser.parse_header()?;
    
    // Load weights
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model weights loaded successfully");
    
    let load_time = start_time.elapsed();
    println!("📊 Model loading time: {:?}", load_time);
    println!("📊 Model size: {} parameters", config.vocab_size * config.hidden_size);
    
    Ok(())
}
EOF
    
    # Compile and run test
    local test_dir="/tmp/test-loading-$model_name"
    mkdir -p "$test_dir"
    
    cat > "$test_dir/Cargo.toml" << EOF
[package]
name = "test-loading-$model_name"
version = "0.1.0"
edition = "2021"

[dependencies]
wasm-chord-core = { path = "/home/puneet/wasm-chord/crates/wasm-chord-core" }
wasm-chord-runtime = { path = "/home/puneet/wasm-chord/crates/wasm-chord-runtime" }
EOF
    
    cp "$test_script" "$test_dir/main.rs"
    
    # Run test with timeout
    if timeout "$TIMEOUT_SECONDS" cargo run --manifest-path "$test_dir/Cargo.toml" > "$TEST_RESULTS_DIR/$test_name.log" 2>&1; then
        log_success "Model loading test passed for $model_name"
        echo "PASS" > "$TEST_RESULTS_DIR/$test_name.result"
    else
        log_error "Model loading test failed for $model_name"
        echo "FAIL" > "$TEST_RESULTS_DIR/$test_name.result"
    fi
    
    # Cleanup
    rm -rf "$test_dir" "$test_script"
}

# Test model inference
test_model_inference() {
    local model_name="$1"
    local model_path="$MODELS_DIR/$model_name.gguf"
    local prompt_type="$2"
    local prompt="${TEST_PROMPTS[$prompt_type]}"
    local test_name="inference-$model_name-$prompt_type"
    
    log_info "Testing inference for $model_name with $prompt_type prompt..."
    
    # Create test script
    local test_script="/tmp/test-inference-$model_name-$prompt_type.rs"
    cat > "$test_script" << EOF
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig, GenerationConfig};
use std::io::Cursor;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "$model_path";
    let prompt = "$prompt";
    
    // Load model (simplified version)
    let model_data = std::fs::read(model_path)?;
    let cursor = Cursor::new(model_data);
    let mut parser = GGUFParser::new(cursor);
    
    let meta = parser.parse_header()?;
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);
    
    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }
    
    let cursor = Cursor::new(model_data);
    let mut parser = GGUFParser::new(cursor);
    parser.parse_header()?;
    
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    
    // Test inference
    let config = GenerationConfig {
        max_tokens: $MAX_TOKENS,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
    };
    
    let start_time = Instant::now();
    let result = model.generate(prompt, &tokenizer, &config)?;
    let inference_time = start_time.elapsed();
    
    println!("✅ Inference completed successfully");
    println!("📊 Inference time: {:?}", inference_time);
    println!("📊 Generated text length: {} characters", result.len());
    println!("📊 Tokens per second: {:.2}", result.len() as f64 / inference_time.as_secs_f64());
    println!("📊 Generated text: {}", result);
    
    Ok(())
}
EOF
    
    # Compile and run test
    local test_dir="/tmp/test-inference-$model_name-$prompt_type"
    mkdir -p "$test_dir"
    
    cat > "$test_dir/Cargo.toml" << EOF
[package]
name = "test-inference-$model_name-$prompt_type"
version = "0.1.0"
edition = "2021"

[dependencies]
wasm-chord-core = { path = "/home/puneet/wasm-chord/crates/wasm-chord-core" }
wasm-chord-runtime = { path = "/home/puneet/wasm-chord/crates/wasm-chord-runtime" }
EOF
    
    cp "$test_script" "$test_dir/main.rs"
    
    # Run test with timeout
    if timeout "$TIMEOUT_SECONDS" cargo run --manifest-path "$test_dir/Cargo.toml" > "$TEST_RESULTS_DIR/$test_name.log" 2>&1; then
        log_success "Inference test passed for $model_name ($prompt_type)"
        echo "PASS" > "$TEST_RESULTS_DIR/$test_name.result"
    else
        log_error "Inference test failed for $model_name ($prompt_type)"
        echo "FAIL" > "$TEST_RESULTS_DIR/$test_name.result"
    fi
    
    # Cleanup
    rm -rf "$test_dir" "$test_script"
}

# Test memory usage
test_memory_usage() {
    local model_name="$1"
    local model_path="$MODELS_DIR/$model_name.gguf"
    local test_name="memory-$model_name"
    
    log_info "Testing memory usage for $model_name..."
    
    # Monitor memory usage during model loading
    local memory_log="$TEST_RESULTS_DIR/$test_name-memory.log"
    
    # Start memory monitoring
    (
        while true; do
            echo "$(date): $(free -m | grep '^Mem:' | awk '{print $3}') MB used" >> "$memory_log"
            sleep 1
        done
    ) &
    local monitor_pid=$!
    
    # Run model loading test
    test_model_loading "$model_name"
    
    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true
    
    # Analyze memory usage
    local max_memory=$(grep -o '[0-9]* MB used' "$memory_log" | grep -o '[0-9]*' | sort -n | tail -1)
    log_info "Peak memory usage: ${max_memory}MB"
    
    if [[ $max_memory -gt 8000 ]]; then
        log_warning "High memory usage detected (${max_memory}MB)"
    else
        log_success "Memory usage acceptable (${max_memory}MB)"
    fi
}

# Test GPU acceleration
test_gpu_acceleration() {
    local model_name="$1"
    local model_path="$MODELS_DIR/$model_name.gguf"
    local test_name="gpu-$model_name"
    
    log_info "Testing GPU acceleration for $model_name..."
    
    # Check if GPU is available
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "No NVIDIA GPU available, skipping GPU test"
        echo "SKIP" > "$TEST_RESULTS_DIR/$test_name.result"
        return
    fi
    
    # Create GPU test script
    local test_script="/tmp/test-gpu-$model_name.rs"
    cat > "$test_script" << EOF
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig, GenerationConfig};
use std::io::Cursor;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "$model_path";
    
    // Load model
    let model_data = std::fs::read(model_path)?;
    let cursor = Cursor::new(model_data);
    let mut parser = GGUFParser::new(cursor);
    
    let meta = parser.parse_header()?;
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);
    
    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }
    
    let cursor = Cursor::new(model_data);
    let mut parser = GGUFParser::new(cursor);
    parser.parse_header()?;
    
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    
    // Test GPU initialization
    #[cfg(feature = "cuda")]
    {
        model.init_candle_gpu()?;
        println!("✅ GPU backend initialized successfully");
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️ CUDA feature not enabled, skipping GPU test");
        return Ok(());
    }
    
    // Test inference with GPU
    let config = GenerationConfig {
        max_tokens: $MAX_TOKENS,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
    };
    
    let start_time = Instant::now();
    let result = model.generate("Test prompt for GPU acceleration", &tokenizer, &config)?;
    let inference_time = start_time.elapsed();
    
    println!("✅ GPU inference completed successfully");
    println!("📊 GPU inference time: {:?}", inference_time);
    println!("📊 Tokens per second: {:.2}", result.len() as f64 / inference_time.as_secs_f64());
    
    Ok(())
}
EOF
    
    # Compile and run test
    local test_dir="/tmp/test-gpu-$model_name"
    mkdir -p "$test_dir"
    
    cat > "$test_dir/Cargo.toml" << EOF
[package]
name = "test-gpu-$model_name"
version = "0.1.0"
edition = "2021"

[dependencies]
wasm-chord-core = { path = "/home/puneet/wasm-chord/crates/wasm-chord-core" }
wasm-chord-runtime = { path = "/home/puneet/wasm-chord/crates/wasm-chord-runtime" }
EOF
    
    cp "$test_script" "$test_dir/main.rs"
    
    # Run test with CUDA feature
    if timeout "$TIMEOUT_SECONDS" cargo run --features cuda --manifest-path "$test_dir/Cargo.toml" > "$TEST_RESULTS_DIR/$test_name.log" 2>&1; then
        log_success "GPU test passed for $model_name"
        echo "PASS" > "$TEST_RESULTS_DIR/$test_name.result"
    else
        log_error "GPU test failed for $model_name"
        echo "FAIL" > "$TEST_RESULTS_DIR/$test_name.result"
    fi
    
    # Cleanup
    rm -rf "$test_dir" "$test_script"
}

# Generate test report
generate_report() {
    local report_file="large-model-test-report.md"
    
    log_info "Generating test report..."
    
    cat > "$report_file" << EOF
# Large Model Test Report

**Generated:** $(date)
**Test Environment:** $(uname -a)
**Total Memory:** $(free -h | grep '^Mem:' | awk '{print $2}')
**Available Memory:** $(free -h | grep '^Mem:' | awk '{print $7}')

## Test Summary

| Model | Loading | Inference | Memory | GPU | Status |
|-------|---------|-----------|--------|-----|--------|
EOF
    
    # Add test results
    for model_name in "${!LARGE_MODELS[@]}"; do
        local loading_result="N/A"
        local inference_result="N/A"
        local memory_result="N/A"
        local gpu_result="N/A"
        local overall_status="UNKNOWN"
        
        if [[ -f "$TEST_RESULTS_DIR/model-loading-$model_name.result" ]]; then
            loading_result=$(cat "$TEST_RESULTS_DIR/model-loading-$model_name.result")
        fi
        
        if [[ -f "$TEST_RESULTS_DIR/inference-$model_name-general.result" ]]; then
            inference_result=$(cat "$TEST_RESULTS_DIR/inference-$model_name-general.result")
        fi
        
        if [[ -f "$TEST_RESULTS_DIR/memory-$model_name.result" ]]; then
            memory_result=$(cat "$TEST_RESULTS_DIR/memory-$model_name.result")
        fi
        
        if [[ -f "$TEST_RESULTS_DIR/gpu-$model_name.result" ]]; then
            gpu_result=$(cat "$TEST_RESULTS_DIR/gpu-$model_name.result")
        fi
        
        # Determine overall status
        if [[ "$loading_result" == "PASS" && "$inference_result" == "PASS" ]]; then
            overall_status="✅ PASS"
        elif [[ "$loading_result" == "FAIL" || "$inference_result" == "FAIL" ]]; then
            overall_status="❌ FAIL"
        else
            overall_status="⚠️ PARTIAL"
        fi
        
        echo "| $model_name | $loading_result | $inference_result | $memory_result | $gpu_result | $overall_status |" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

## Detailed Results

### Model Loading Tests
EOF
    
    for model_name in "${!LARGE_MODELS[@]}"; do
        if [[ -f "$TEST_RESULTS_DIR/model-loading-$model_name.log" ]]; then
            echo "#### $model_name" >> "$report_file"
            echo '```' >> "$report_file"
            cat "$TEST_RESULTS_DIR/model-loading-$model_name.log" >> "$report_file"
            echo '```' >> "$report_file"
            echo "" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

### Inference Tests
EOF
    
    for model_name in "${!LARGE_MODELS[@]}"; do
        for prompt_type in "${!TEST_PROMPTS[@]}"; do
            if [[ -f "$TEST_RESULTS_DIR/inference-$model_name-$prompt_type.log" ]]; then
                echo "#### $model_name ($prompt_type)" >> "$report_file"
                echo '```' >> "$report_file"
                cat "$TEST_RESULTS_DIR/inference-$model_name-$prompt_type.log" >> "$report_file"
                echo '```' >> "$report_file"
                echo "" >> "$report_file"
            fi
        done
    done
    
    cat >> "$report_file" << EOF

## Recommendations

1. **Memory Requirements**: Ensure at least 8GB RAM for 7B models
2. **GPU Acceleration**: Use CUDA-enabled builds for better performance
3. **Model Selection**: Choose Q4_K_M quantization for best balance of size and quality
4. **Production Deployment**: Test thoroughly with target hardware before deployment

## Conclusion

Large model testing completed successfully! 🎉

EOF
    
    log_success "Test report saved to $report_file"
}

# Main execution
main() {
    echo "🚀 wasm-chord Large Model Test Suite"
    echo "====================================="
    echo ""
    
    # Create results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Check system requirements
    check_system_requirements
    
    echo ""
    log_info "Starting large model testing..."
    echo ""
    
    # Test each model
    for model_name in "${!LARGE_MODELS[@]}"; do
        local model_url="${LARGE_MODELS[$model_name]}"
        
        log_info "Testing model: $model_name"
        echo "=========================================="
        
        # Download model
        if download_model "$model_name" "$model_url"; then
            # Test model loading
            test_model_loading "$model_name"
            
            # Test memory usage
            test_memory_usage "$model_name"
            
            # Test inference with different prompts
            for prompt_type in "${!TEST_PROMPTS[@]}"; do
                test_model_inference "$model_name" "$prompt_type"
            done
            
            # Test GPU acceleration
            test_gpu_acceleration "$model_name"
            
            log_success "Completed testing for $model_name"
        else
            log_error "Failed to download $model_name, skipping tests"
        fi
        
        echo ""
    done
    
    # Generate report
    generate_report
    
    log_success "Large model test suite completed! 🎉"
    echo ""
    log_info "Check large-model-test-report.md for detailed results"
    log_info "Individual test logs available in $TEST_RESULTS_DIR/"
}

# Run main function
main "$@"
