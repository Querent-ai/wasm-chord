#!/usr/bin/env python3
"""
Check what token "Yes" maps to in our tokenizer
"""
import subprocess
import re

def check_yes_token():
    """Check what token 'Yes' maps to"""
    print("🔍 Checking tokenization of 'Yes'")
    
    cmd = ["cargo", "run", "--release", "--manifest-path", "examples/ollama-comparison/Cargo.toml"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Look for the tokenization section
        in_tokenization = False
        for line in output.split('\n'):
            if 'Testing multiple prompts:' in line:
                in_tokenization = True
                continue
            elif in_tokenization and "'Yes'" in line:
                # Parse line like "  'Yes' -> token 3869 ("▁Yes")"
                match = re.search(r"'Yes' -> token (\d+) \("([^"]+)"\)", line)
                if match:
                    token_id, token_text = match.groups()
                    print(f"✅ 'Yes' maps to token {token_id} ('{token_text}')")
                    return int(token_id), token_text
                    
        print("❌ Could not find 'Yes' tokenization")
        return None, None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def check_yes_logit():
    """Check the logit value for the 'Yes' token"""
    print("\n🔍 Checking logit for 'Yes' token")
    
    cmd = ["cargo", "run", "--release", "--manifest-path", "examples/ollama-comparison/Cargo.toml"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Look for the specific token check
        for line in output.split('\n'):
            if 'Checking specific expected tokens:' in line:
                # Look for the next few lines
                lines = output.split('\n')
                for i, l in enumerate(lines):
                    if 'Checking specific expected tokens:' in l:
                        # Check next few lines for "Yes" or "▁Yes"
                        for j in range(i+1, min(i+10, len(lines))):
                            if 'Yes' in lines[j] or '▁Yes' in lines[j]:
                                print(f"✅ Found Yes token info: {lines[j]}")
                                return lines[j]
                        break
                        
        print("❌ Could not find 'Yes' token logit")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    token_id, token_text = check_yes_token()
    if token_id:
        logit_info = check_yes_logit()
        if logit_info:
            print(f"\n📊 Summary:")
            print(f"   'Yes' -> token {token_id} ('{token_text}')")
            print(f"   Logit info: {logit_info}")
