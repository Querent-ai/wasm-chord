#!/bin/bash

echo "=== wasm-chord tokens (temperature=0, greedy) ==="
echo "Prompt: 'Hello'"
echo "Tokens: [1, 15043, 9957, 2749, 2749]"
echo "Output: Helloessenarrarr"
echo ""

echo "=== ollama tokens (same model: tinyllama:latest) ==="
echo "Running ollama to get first few tokens..."
echo "Note: ollama adds chat formatting, so tokens will differ"
echo "We need to compare the BASE model without chat wrapping"
echo ""

# Try to get raw model output if possible
# Note: ollama might not expose raw token generation
echo "Checking ollama model info..."
ollama show tinyllama:latest 2>&1 | head -20
