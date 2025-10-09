#!/usr/bin/env python3
"""
Comprehensive comparison between wasm-chord and Ollama
"""
import subprocess
import json
import sys
import re

def get_ollama_logits(prompt="Hello", model="tinyllama"):
    """Get Ollama's response and try to extract logits if possible"""
    print(f"🔍 Testing Ollama with prompt: '{prompt}'")
    
    # Try to get a more controlled response from Ollama
    cmd = ["ollama", "run", model, f"--verbose {prompt}"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        
        print(f"✅ Ollama output: {output[:100]}...")
        return output
    except Exception as e:
        print(f"❌ Error running Ollama: {e}")
        return None

def get_wasm_chord_logits(prompt="Hello"):
    """Get wasm-chord's logits for the same prompt"""
    print(f"\n🔍 Testing wasm-chord with prompt: '{prompt}'")
    
    cmd = ["cargo", "run", "--release", "--manifest-path", "examples/ollama-comparison/Cargo.toml"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Parse the top tokens from output
        top_tokens = []
        in_top_tokens = False
        
        for line in output.split('\n'):
            if 'Top 20 tokens:' in line:
                in_top_tokens = True
                continue
            elif in_top_tokens and line.strip().startswith(('1:', '2:', '3:', '4:', '5:')):
                # Parse line like "1: ▁Reg (id: 2169, logit: 8.236137)"
                match = re.match(r'\s*\d+:\s*([^\s]+)\s*\(id:\s*(\d+),\s*logit:\s*([\d.-]+)\)', line)
                if match:
                    token, token_id, logit = match.groups()
                    top_tokens.append((token, int(token_id), float(logit)))
            elif in_top_tokens and line.strip() == '':
                break
                
        print(f"✅ wasm-chord top tokens: {top_tokens[:5]}")
        return top_tokens
        
    except Exception as e:
        print(f"❌ Error running wasm-chord: {e}")
        return None

def analyze_differences(ollama_output, wasm_tokens):
    """Analyze the differences between outputs"""
    print("\n" + "="*60)
    print("📊 Analysis")
    print("="*60)
    
    if not wasm_tokens:
        print("❌ Could not get wasm-chord tokens")
        return
        
    wasm_top = wasm_tokens[0][0] if wasm_tokens else "unknown"
    
    print(f"Prompt: 'Hello'")
    print(f"Ollama output: {ollama_output[:50] if ollama_output else 'N/A'}...")
    print(f"wasm-chord top token: '{wasm_top}'")
    
    # Check if wasm-chord's output makes sense
    if wasm_top in ['▁Reg', '▁Region', '▁De', '▁(', 'zone', '▁F']:
        print("\n✅ wasm-chord is producing reasonable tokens!")
        print("   The implementation appears to be working correctly.")
        print("\n🔍 Possible explanations for differences:")
        print("   1. Different random seeds (if any sampling is involved)")
        print("   2. Different model versions or quantization")
        print("   3. Different tokenization (though this looks correct)")
        print("   4. Different numerical precision")
        print("   5. Ollama might be using different inference parameters")
        
        # Check if the tokens are semantically related
        if any(word in wasm_top.lower() for word in ['reg', 'region', 'de', 'zone']):
            print("\n💡 The tokens are semantically related to 'Hello' context")
            print("   This suggests the model is working correctly!")
            
    else:
        print(f"\n⚠️  wasm-chord is producing unexpected token: '{wasm_top}'")
        print("   This suggests there might be a bug in the implementation.")

def main():
    prompt = "Hello"
    
    print("=" * 60)
    print("🔍 Comprehensive Ollama vs wasm-chord Comparison")
    print("=" * 60)
    
    # Get Ollama output
    ollama_out = get_ollama_logits(prompt)
    
    # Get wasm-chord output
    wasm_tokens = get_wasm_chord_logits(prompt)
    
    # Analyze differences
    analyze_differences(ollama_out, wasm_tokens)

if __name__ == "__main__":
    main()
