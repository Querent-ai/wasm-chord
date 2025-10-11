#!/bin/bash
# Download TinyLLaMA model for CI/testing

set -e

MODEL_DIR="${MODEL_DIR:-/tmp/wasm-chord-models}"
MODEL_NAME="tinyllama-1.1b.Q4_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# Hugging Face URL
HF_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "📥 Downloading TinyLLaMA model for testing..."
echo "   Target: $MODEL_PATH"

# Create directory
mkdir -p "$MODEL_DIR"

# Check if already downloaded
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already exists at $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    exit 0
fi

# Download with wget or curl
if command -v wget &> /dev/null; then
    echo "📥 Using wget to download..."
    wget -O "$MODEL_PATH" "$HF_URL"
elif command -v curl &> /dev/null; then
    echo "📥 Using curl to download..."
    curl -L -o "$MODEL_PATH" "$HF_URL"
else
    echo "❌ Error: Neither wget nor curl found"
    exit 1
fi

echo "✅ Download complete!"
ls -lh "$MODEL_PATH"

# Verify it's a valid GGUF file
if file "$MODEL_PATH" | grep -q "data"; then
    echo "✅ File appears to be valid"
else
    echo "⚠️  Warning: File might not be a valid GGUF file"
fi

echo ""
echo "Model ready at: $MODEL_PATH"
echo "Set environment variable: export WASM_CHORD_MODEL_PATH=$MODEL_PATH"
