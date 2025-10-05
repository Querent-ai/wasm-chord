#!/bin/bash

# Test script to compare wasm-chord with ollama token-by-token

echo "Testing with simple prompt:"
PROMPT="Hello"

echo "=== Ollama output ==="
echo "$PROMPT" | ollama run tinyllama:latest 2>&1 | head -5

echo ""
echo "=== wasm-chord output ==="
# We'll add this after fixing
