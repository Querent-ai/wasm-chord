#!/usr/bin/env python3
"""
Simple test to compare our implementation with a known working reference
"""
import subprocess
import sys

def test_simple_generation():
    """Test simple generation with debug output"""
    print("🔍 Testing simple generation with debug output")
    
    # Run our implementation with debug flags
    cmd = [
        "cargo", "run", "--release", 
        "--manifest-path", "examples/ollama-comparison/Cargo.toml"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ Error running wasm-chord: {result.stderr}")
            return False
            
        output = result.stdout
        print("✅ wasm-chord output:")
        print(output)
        
        # Look for the top token
        for line in output.split('\n'):
            if 'Our greedy output:' in line:
                token = line.split("'")[1]
                print(f"🎯 Top token: '{token}'")
                
                # Check if this is reasonable
                if token in ['▁Reg', '▁Region', '▁De', '▁(', 'zone', '▁F']:
                    print("✅ Getting reasonable tokens")
                    return True
                else:
                    print(f"⚠️  Unexpected token: '{token}'")
                    return False
                    
        print("❌ Could not find top token in output")
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout running wasm-chord")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_generation()
    sys.exit(0 if success else 1)
