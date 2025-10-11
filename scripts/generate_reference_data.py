#!/usr/bin/env python3
"""
Generate reference data from llama.cpp for integration testing.
This script captures deterministic outputs that our wasm-chord implementation should match.
"""

import subprocess
import json
import sys
from pathlib import Path

def run_llamacpp(prompt, num_tokens=5, temp=0):
    """Run llama.cpp and capture the output tokens and logits."""
    cmd = [
        "/home/puneet/wasm-chord/models/llama.cpp/build/bin/llama-cli",
        "-m", "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf",
        "-p", prompt,
        "-n", str(num_tokens),
        "--temp", str(temp),
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

def main():
    """Generate reference data for multiple test cases."""
    
    test_cases = [
        {"prompt": "Hello", "num_tokens": 5},
        {"prompt": "The", "num_tokens": 5},
        {"prompt": "Once upon a time", "num_tokens": 5},
        {"prompt": "Paris is", "num_tokens": 5},
    ]
    
    reference_data = {}
    
    for test_case in test_cases:
        print(f"Generating reference for: '{test_case['prompt']}'")
        result = run_llamacpp(test_case["prompt"], test_case["num_tokens"])
        
        if result:
            reference_data[test_case["prompt"]] = result
            print(f"✅ Generated {len(result['tokens'])} tokens for '{test_case['prompt']}'")
        else:
            print(f"❌ Failed to generate reference for '{test_case['prompt']}'")
    
    # Save reference data
    output_file = Path("/home/puneet/wasm-chord/reference_data.json")
    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)
    
    print(f"\n📁 Reference data saved to: {output_file}")
    print(f"📊 Generated references for {len(reference_data)} test cases")
    
    # Print summary
    for prompt, data in reference_data.items():
        tokens_text = [t["text"] for t in data["tokens"]]
        print(f"  '{prompt}' -> {tokens_text}")

if __name__ == "__main__":
    main()
