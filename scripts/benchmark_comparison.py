#!/usr/bin/env python3
"""
Benchmark Comparison Script
This script compares wasm-chord outputs with llama.cpp outputs to verify our benchmarks are correct.
"""

import subprocess
import json
import sys
from pathlib import Path

def run_llamacpp(prompt, num_tokens=5, temp=0):
    """Run llama.cpp and capture the output tokens."""
    cmd = [
        "/home/puneet/wasm-chord/models/llama.cpp/build/bin/llama-cli",
        "-m", "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf",
        "-p", prompt,
        "-n", str(num_tokens),
        "--temp", str(temp),
        "--no-conversation",
        "--log-verbose"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Error running llama.cpp: {result.stderr}")
            return None
        
        # Parse the output to extract tokens
        output_lines = result.stdout.split('\n')
        tokens = []
        
        # Look for token generation lines
        for line in output_lines:
            if "eval:" in line and "[" in line and "]" in line:
                # Extract tokens from lines like: eval: [ 'token':id ]
                parts = line.split('[')[1].split(']')[0]
                if parts.strip():
                    # Parse individual tokens
                    token_parts = parts.split(',')
                    for part in token_parts:
                        part = part.strip()
                        if "':" in part:
                            token_text = part.split("':")[0].strip().strip("'")
                            token_id = part.split("':")[1].strip()
                            tokens.append({"text": token_text, "id": int(token_id)})
        
        return {
            "prompt": prompt,
            "tokens": tokens,
            "raw_output": result.stdout
        }
    except subprocess.TimeoutExpired:
        print("llama.cpp timed out")
        return None
    except Exception as e:
        print(f"Error running llama.cpp: {e}")
        return None

def run_wasm_chord(prompt):
    """Run wasm-chord ollama-comparison example."""
    cmd = [
        "cargo", "run", "--example", "ollama-comparison"
    ]
    
    try:
        # Set environment variable to override the prompt
        env = {"WASM_CHORD_PROMPT": prompt}
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        
        if result.returncode != 0:
            print(f"Error running wasm-chord: {result.stderr}")
            return None
        
        # Parse the output to extract top logits
        output_lines = result.stdout.split('\n')
        top_logits = []
        
        for line in output_lines:
            if ":" in line and "id:" in line and "logit:" in line:
                # Parse lines like: "1: , (id: 29892, logit: 10.308499)"
                try:
                    parts = line.split("(")[1].split(")")[0]
                    id_part = parts.split("id:")[1].split(",")[0].strip()
                    logit_part = parts.split("logit:")[1].strip()
                    
                    token_id = int(id_part)
                    logit = float(logit_part)
                    
                    # Extract token text
                    token_text = line.split(":")[1].split("(")[0].strip()
                    
                    top_logits.append({
                        "token_id": token_id,
                        "text": token_text,
                        "logit": logit
                    })
                except (IndexError, ValueError):
                    continue
        
        return {
            "prompt": prompt,
            "top_logits": top_logits,
            "raw_output": result.stdout
        }
    except subprocess.TimeoutExpired:
        print("wasm-chord timed out")
        return None
    except Exception as e:
        print(f"Error running wasm-chord: {e}")
        return None

def compare_outputs(llamacpp_result, wasm_chord_result, tolerance=0.01):
    """Compare llama.cpp and wasm-chord outputs."""
    print(f"\n🔍 Comparing outputs for prompt: '{llamacpp_result['prompt']}'")
    
    # Compare top logits
    print("\n📊 Top Logits Comparison:")
    print("Rank | Token ID | Token Text | llama.cpp Logit | wasm-chord Logit | Diff")
    print("-" * 80)
    
    matches = 0
    total = 0
    
    # Get top 5 logits from wasm-chord
    wasm_logits = sorted(wasm_chord_result["top_logits"], key=lambda x: x["logit"], reverse=True)[:5]
    
    for i, wasm_logit in enumerate(wasm_logits):
        token_id = wasm_logit["token_id"]
        token_text = wasm_logit["text"]
        wasm_logit_val = wasm_logit["logit"]
        
        # Find corresponding llama.cpp logit (we'll use our reference data for now)
        # In a real comparison, we'd extract this from llama.cpp output
        llama_logit_val = 0.0  # Placeholder
        
        diff = abs(wasm_logit_val - llama_logit_val)
        match = diff <= tolerance
        
        print(f"{i+1:4} | {token_id:8} | {token_text:10} | {llama_logit_val:13.6f} | {wasm_logit_val:15.6f} | {diff:.6f}")
        
        if match:
            matches += 1
        total += 1
    
    print(f"\n✅ Matches: {matches}/{total} (tolerance: {tolerance})")
    
    # Compare generated tokens
    print(f"\n🎯 Generated Tokens:")
    print("llama.cpp tokens:", [t["text"] for t in llamacpp_result["tokens"]])
    print("wasm-chord top token:", wasm_logits[0]["text"] if wasm_logits else "None")
    
    return matches == total

def main():
    """Main comparison function."""
    print("🔍 Benchmark Comparison: llama.cpp vs wasm-chord")
    print("=" * 60)
    
    test_prompts = ["Hello", "The", "Once upon a time"]
    
    for prompt in test_prompts:
        print(f"\n🧪 Testing prompt: '{prompt}'")
        
        # Run llama.cpp
        print("Running llama.cpp...")
        llamacpp_result = run_llamacpp(prompt, 5, 0)
        
        if not llamacpp_result:
            print(f"❌ Failed to get llama.cpp results for '{prompt}'")
            continue
        
        # Run wasm-chord
        print("Running wasm-chord...")
        wasm_chord_result = run_wasm_chord(prompt)
        
        if not wasm_chord_result:
            print(f"❌ Failed to get wasm-chord results for '{prompt}'")
            continue
        
        # Compare outputs
        matches = compare_outputs(llamacpp_result, wasm_chord_result)
        
        if matches:
            print(f"✅ Benchmark comparison PASSED for '{prompt}'")
        else:
            print(f"❌ Benchmark comparison FAILED for '{prompt}'")
    
    print("\n🎉 Benchmark comparison complete!")

if __name__ == "__main__":
    main()
