#!/usr/bin/env python3
"""
Get ground truth logits from llama.cpp Python bindings

This script uses llama-cpp-python to get the logits for comparison
with our wasm-chord implementation.
"""

try:
    from llama_cpp import Llama
    import numpy as np
except ImportError:
    print("❌ llama-cpp-python not installed")
    print("   Install with: pip install llama-cpp-python")
    exit(1)

model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf"
prompt = "Hello"

print("🔍 Loading model with llama.cpp...")
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    verbose=False,
)

print(f"✅ Model loaded")
print(f"🎯 Testing prompt: '{prompt}'")

# Tokenize
tokens = llm.tokenize(prompt.encode('utf-8'))
print(f"📝 Tokens: {tokens}")

# Get logits for the first token
llm.reset()
llm.eval(tokens)

# Get the logits from the model
logits = llm._scores[-1]  # Last token's logits
logits_array = np.array(logits)

print(f"📊 Logits shape: {logits_array.shape}")

# Get top 20 tokens
top_indices = np.argsort(logits_array)[::-1][:20]

print("\n🏆 Top 20 tokens (from llama.cpp):")
for i, idx in enumerate(top_indices):
    token = llm.detokenize([idx]).decode('utf-8', errors='ignore')
    print(f"  {i+1}: token {idx} = {logits_array[idx]:.6f} ({repr(token)})")

# Check specific token "global" (ID 10945 in our implementation)
if 10945 < len(logits_array):
    print(f"\n🔍 Token 10945 ('global' in wasm-chord): {logits_array[10945]:.6f}")

# Save logits for detailed comparison
np.save('/tmp/llama_cpp_logits.npy', logits_array)
print("\n💾 Logits saved to /tmp/llama_cpp_logits.npy")
print(f"   Total vocab size: {len(logits_array)}")
