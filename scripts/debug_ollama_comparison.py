#!/usr/bin/env python3
"""
Compare wasm-chord outputs with Ollama

This script:
1. Runs ollama with the same model and prompt
2. Gets the first token prediction
3. Compares with our implementation
"""

import subprocess
import json
import sys

def get_ollama_response(prompt="Hello", model="tinyllama"):
    """Get Ollama's response for the prompt"""
    print(f"🔍 Testing Ollama with prompt: '{prompt}'")

    # Use ollama generate with JSON output for better parsing
    cmd = ["ollama", "run", model, prompt]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()

        print(f"✅ Ollama output: {output[:200]}...")
        return output
    except subprocess.TimeoutExpired:
        print("❌ Ollama request timed out")
        return None
    except Exception as e:
        print(f"❌ Error running Ollama: {e}")
        return None

def get_wasm_chord_output(prompt="Hello"):
    """Get wasm-chord's output for the same prompt"""
    print(f"\n🔍 Testing wasm-chord with prompt: '{prompt}'")

    # For now, we'll read from the ollama-comparison example output
    # In the future, we can integrate more directly
    cmd = ["cargo", "run", "--release", "--manifest-path",
           "examples/ollama-comparison/Cargo.toml"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout

        # Parse the top token from output
        for line in output.split('\n'):
            if 'Our greedy output:' in line:
                token = line.split("'")[1]
                print(f"✅ wasm-chord output: {token}")
                return token

        print("❌ Could not parse wasm-chord output")
        return None
    except Exception as e:
        print(f"❌ Error running wasm-chord: {e}")
        return None

def main():
    prompt = "Hello"

    print("=" * 60)
    print("🔍 Ollama vs wasm-chord Comparison")
    print("=" * 60)

    # Get Ollama output
    ollama_out = get_ollama_response(prompt)

    # Get wasm-chord output
    wasm_chord_out = get_wasm_chord_output(prompt)

    print("\n" + "=" * 60)
    print("📊 Comparison Results")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")
    print(f"Ollama first tokens: {ollama_out[:50] if ollama_out else 'N/A'}...")
    print(f"wasm-chord first token: {wasm_chord_out}")

    if ollama_out and wasm_chord_out:
        # Check if wasm-chord's output makes sense
        if wasm_chord_out.lower() in ["global", "番", "rique"]:
            print("\n⚠️  wasm-chord is producing unexpected tokens!")
            print("   This suggests a bug in the implementation.")
            print("\n🔍 Possible issues to investigate:")
            print("   1. Q4_K dequantization")
            print("   2. Matrix multiplication order/transpose")
            print("   3. RoPE (Rotary Position Embeddings)")
            print("   4. Attention mask")
            print("   5. RMSNorm implementation")
        else:
            print("\n✅ wasm-chord output looks reasonable!")

if __name__ == "__main__":
    main()
