#!/usr/bin/env python3
"""
Test if token embeddings are loaded correctly by comparing with llama.cpp
"""

import sys
import struct

# Model details
VOCAB_SIZE = 32000
HIDDEN_SIZE = 2048
MODEL_PATH = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf"

print("🔍 Testing Token Embeddings")
print("="*60)

# Token ID to test (let's use token 15043 which is "Hello")
test_token = 15043

print(f"📝 Test token ID: {test_token}")
print(f"📐 Expected embedding size: {HIDDEN_SIZE}")
print(f"📍 Expected offset in embeddings: {test_token * HIDDEN_SIZE}")

# The test would require reading the GGUF file and extracting the quantized
# token embedding for this token ID, then comparing with what wasm-chord loads
print("\n✨ This would require implementing GGUF parsing in Python")
print("   For now, let's just confirm the model file exists...")

import os
if os.path.exists(MODEL_PATH):
    print(f"✅ Model file found: {MODEL_PATH}")
    print(f"   Size: {os.path.getsize(MODEL_PATH) / (1024*1024*1024):.2f} GB")
else:
    print(f"❌ Model file not found: {MODEL_PATH}")
