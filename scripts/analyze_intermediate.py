#!/usr/bin/env python3
"""
Analyze wasm-chord intermediate values to look for obvious issues
"""

def analyze_intermediate_values():
    """Analyze the intermediate values from wasm-chord debug trace"""
    
    print("🔍 WASM-CHORD INTERMEDIATE VALUE ANALYSIS")
    print("=" * 50)
    
    # Values from debug trace
    emb = [-0.011065, 0.001582, -0.007904, -0.020551, -0.004742]
    l0_out = [-0.015798, 0.004437, -0.014060, -0.017191, 0.006112]
    l21_out = [-0.581552, -0.488692, -0.266416, 0.854834, 1.153147]
    hidden = [-1.251323, -0.995949, -0.580238, 1.899160, 2.471138]
    
    print("📊 INTERMEDIATE VALUES:")
    print(f"   EMB[0:5]:     {emb}")
    print(f"   L0_OUT[0:5]:  {l0_out}")
    print(f"   L21_OUT[0:5]: {l21_out}")
    print(f"   HIDDEN[0:5]:  {hidden}")
    
    # Analyze magnitude progression
    print(f"\n📈 MAGNITUDE PROGRESSION:")
    emb_mag = max(abs(x) for x in emb)
    l0_mag = max(abs(x) for x in l0_out)
    l21_mag = max(abs(x) for x in l21_out)
    hidden_mag = max(abs(x) for x in hidden)
    
    print(f"   EMB max magnitude:     {emb_mag:.6f}")
    print(f"   L0_OUT max magnitude:  {l0_mag:.6f}")
    print(f"   L21_OUT max magnitude: {l21_mag:.6f}")
    print(f"   HIDDEN max magnitude:  {hidden_mag:.6f}")
    
    # Check if magnitudes are reasonable
    print(f"\n🔍 MAGNITUDE ANALYSIS:")
    if emb_mag > 1.0:
        print(f"   ⚠️  EMB magnitude ({emb_mag:.3f}) seems high for embeddings")
    else:
        print(f"   ✅ EMB magnitude ({emb_mag:.3f}) looks reasonable")
    
    if l0_mag > 1.0:
        print(f"   ⚠️  L0_OUT magnitude ({l0_mag:.3f}) seems high")
    else:
        print(f"   ✅ L0_OUT magnitude ({l0_mag:.3f}) looks reasonable")
    
    if l21_mag > 2.0:
        print(f"   ⚠️  L21_OUT magnitude ({l21_mag:.3f}) seems high")
    else:
        print(f"   ✅ L21_OUT magnitude ({l21_mag:.3f}) looks reasonable")
    
    if hidden_mag > 5.0:
        print(f"   ⚠️  HIDDEN magnitude ({hidden_mag:.3f}) seems high")
    else:
        print(f"   ✅ HIDDEN magnitude ({hidden_mag:.3f}) looks reasonable")
    
    # Check for NaN or Inf values
    print(f"\n🔍 VALUE VALIDITY:")
    all_values = emb + l0_out + l21_out + hidden
    
    has_nan = any(x != x for x in all_values)  # NaN check
    has_inf = any(abs(x) == float('inf') for x in all_values)
    
    if has_nan:
        print(f"   🚨 FOUND NaN VALUES!")
    else:
        print(f"   ✅ No NaN values")
    
    if has_inf:
        print(f"   🚨 FOUND Inf VALUES!")
    else:
        print(f"   ✅ No Inf values")
    
    # Check for systematic patterns
    print(f"\n🔍 PATTERN ANALYSIS:")
    
    # Check if values are too small (underflow)
    min_abs = min(abs(x) for x in all_values if x != 0)
    if min_abs < 1e-6:
        print(f"   ⚠️  Very small values detected (min: {min_abs:.2e})")
    else:
        print(f"   ✅ Values seem to have reasonable scale")
    
    # Check if values are too large (overflow)
    max_abs = max(abs(x) for x in all_values)
    if max_abs > 100:
        print(f"   ⚠️  Very large values detected (max: {max_abs:.2e})")
    else:
        print(f"   ✅ Values seem to have reasonable scale")
    
    # Analyze the progression from EMB to HIDDEN
    print(f"\n📊 PROGRESSION ANALYSIS:")
    
    # Check if there's a reasonable progression
    emb_range = max(emb) - min(emb)
    l0_range = max(l0_out) - min(l0_out)
    l21_range = max(l21_out) - min(l21_out)
    hidden_range = max(hidden) - min(hidden)
    
    print(f"   EMB range:     {emb_range:.6f}")
    print(f"   L0_OUT range:  {l0_range:.6f}")
    print(f"   L21_OUT range: {l21_range:.6f}")
    print(f"   HIDDEN range:  {hidden_range:.6f}")
    
    # Check if the range increases reasonably
    ranges = [emb_range, l0_range, l21_range, hidden_range]
    increasing = all(ranges[i] <= ranges[i+1] for i in range(len(ranges)-1))
    
    if increasing:
        print(f"   ✅ Range increases progressively (expected for transformer)")
    else:
        print(f"   ⚠️  Range doesn't increase progressively")
    
    # Check for specific issues
    print(f"\n🎯 SPECIFIC ISSUE CHECKS:")
    
    # Check if all values are zero (would indicate complete failure)
    all_zero = all(x == 0 for x in all_values)
    if all_zero:
        print(f"   🚨 ALL VALUES ARE ZERO! Complete computation failure!")
    else:
        print(f"   ✅ Not all values are zero")
    
    # Check if values are identical (would indicate no computation)
    emb_unique = len(set(emb))
    l0_unique = len(set(l0_out))
    l21_unique = len(set(l21_out))
    hidden_unique = len(set(hidden))
    
    print(f"   EMB unique values:     {emb_unique}/5")
    print(f"   L0_OUT unique values:  {l0_unique}/5")
    print(f"   L21_OUT unique values: {l21_unique}/5")
    print(f"   HIDDEN unique values:  {hidden_unique}/5")
    
    if emb_unique == 1:
        print(f"   🚨 EMB values are all identical!")
    if l0_unique == 1:
        print(f"   🚨 L0_OUT values are all identical!")
    if l21_unique == 1:
        print(f"   🚨 L21_OUT values are all identical!")
    if hidden_unique == 1:
        print(f"   🚨 HIDDEN values are all identical!")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   • Intermediate values look numerically reasonable")
    print(f"   • No obvious NaN/Inf issues")
    print(f"   • Magnitudes seem appropriate for transformer layers")
    print(f"   • Values show expected progression through layers")
    print(f"   • The issue is likely in the computation logic, not numerical overflow")
    
    print(f"\n🔍 NEXT STEPS:")
    print(f"   1. Need to compare with llama.cpp intermediate values")
    print(f"   2. Check if divergence starts at embeddings or later")
    print(f"   3. Verify weight loading and quantization")
    print(f"   4. Check attention computation specifically")

if __name__ == "__main__":
    analyze_intermediate_values()
