#!/bin/bash

echo "🧪 Testing all TinyLlama models with WASM-Chord"
echo "================================================"

models=(
    "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
    "models/tinyllama-1.1b.Q4_0.gguf" 
    "models/tinyllama-1.1b.Q4_K_M.gguf"
    "models/tinyllama-q4km.gguf"
    "models/tinyllama-q8.gguf"
)

for model in "${models[@]}"; do
    echo ""
    echo "📂 Testing: $(basename "$model")"
    echo "----------------------------------------"
    
    # Update the model path in the example
    sed -i "s|let model_path = \".*\"|let model_path = \"$model\"|" examples/simple-generation/main.rs
    
    # Build and test
    cargo build --release -q 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Build successful"
        timeout 30 ./target/release/simple-generation 2>/dev/null | grep "📝 Result:" | head -1
    else
        echo "❌ Build failed"
    fi
done

echo ""
echo "🔍 Ollama comparison:"
echo "--------------------"
echo "The meaning of life is" | timeout 10 ollama run tinyllama:latest 2>/dev/null | head -1
