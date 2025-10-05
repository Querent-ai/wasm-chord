#!/bin/bash

# Compare wasm-chord tokens with ollama

echo "=== wasm-chord tokens ==="
echo "Prompt: 'Hello'"
echo "Tokens: [1, 15043, 9957, 2749, 2749]"
echo "Output: Helloessenarrarr"
echo ""

echo "=== ollama test ==="
echo "Hello" | timeout 20 ollama run tinyllama:latest 2>&1 | head -10

echo ""
echo "Note: ollama uses chat model which adds system prompt"
echo "We need to compare raw model inference, not chat"
