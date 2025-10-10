#!/usr/bin/env python3
"""
Analyze the divergence between wasm-chord and llama.cpp intermediate values
"""

def analyze_divergence():
    print("🔍 ANALYZING LOGIT DIVERGENCE")
    print("=" * 50)
    
    # Current wasm-chord intermediate values (with fixed weights)
    wasm_chord_values = {
        "EMB": [-0.001300, 0.001904, -0.001941, 0.003827, 0.001263],
        "L0_OUT": [0.001249, -0.002032, 0.000110, 0.008114, 0.005240],
        "L21_OUT": [1.513082, -2.085949, -0.623656, 0.378941, -0.021589],
        "HIDDEN": [3.097959, -3.754161, -1.012926, 0.689021, -0.118847],
    }
    
    # Expected llama.cpp values (we need to estimate these)
    # Token 29892 logit: wasm-chord = 1.40, llama.cpp = 10.82
    # Difference = 9.42 logits
    
    print("\n📊 CURRENT WASM-CHORD INTERMEDIATE VALUES")
    print("-" * 40)
    
    for stage, values in wasm_chord_values.items():
        print(f"{stage}: {values}")
        print(f"  Range: {min(values):.6f} to {max(values):.6f}")
        print(f"  Mean: {sum(values)/len(values):.6f}")
        print(f"  Std: {(sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5:.6f}")
        print()
    
    print("\n🎯 DIVERGENCE ANALYSIS")
    print("-" * 40)
    
    # Analyze the progression
    stages = ["EMB", "L0_OUT", "L21_OUT", "HIDDEN"]
    ranges = []
    
    for stage in stages:
        values = wasm_chord_values[stage]
        range_val = max(values) - min(values)
        ranges.append(range_val)
        print(f"{stage} range: {range_val:.6f}")
    
    print("\n📈 PROGRESSION ANALYSIS")
    print("-" * 40)
    
    # Check if values are growing appropriately
    for i in range(1, len(stages)):
        prev_range = ranges[i-1]
        curr_range = ranges[i]
        growth_factor = curr_range / prev_range if prev_range > 0 else float('inf')
        print(f"{stages[i-1]} -> {stages[i]}: {prev_range:.6f} -> {curr_range:.6f} (x{growth_factor:.2f})")
    
    print("\n🔍 POTENTIAL ISSUES")
    print("-" * 40)
    
    # Check for common issues
    issues = []
    
    # 1. Check if values are too small
    hidden_max = max(wasm_chord_values["HIDDEN"])
    if hidden_max < 5.0:
        issues.append(f"Hidden states too small (max: {hidden_max:.3f})")
    
    # 2. Check if progression is reasonable
    emb_range = ranges[0]
    hidden_range = ranges[-1]
    if hidden_range < emb_range * 10:
        issues.append(f"Hidden states not growing enough (emb: {emb_range:.3f}, hidden: {hidden_range:.3f})")
    
    # 3. Check for numerical issues
    for stage, values in wasm_chord_values.items():
        if any(abs(v) > 100 for v in values):
            issues.append(f"{stage} has very large values")
        if any(abs(v) < 1e-10 and v != 0 for v in values):
            issues.append(f"{stage} has very small non-zero values")
    
    if issues:
        print("❌ Potential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No obvious numerical issues detected")
    
    print("\n🎯 HYPOTHESIS")
    print("-" * 40)
    
    # Based on the 9.42 logit difference, estimate where the issue might be
    logit_diff = 9.42
    
    print(f"Logit difference: {logit_diff:.2f}")
    print(f"This suggests a systematic scaling issue.")
    
    # Estimate what the hidden states should be
    current_hidden_max = max(wasm_chord_values["HIDDEN"])
    estimated_hidden_max = current_hidden_max * (10.82 / 1.40)  # Scale by logit ratio
    
    print(f"\nCurrent hidden max: {current_hidden_max:.3f}")
    print(f"Estimated hidden max for correct logits: {estimated_hidden_max:.3f}")
    print(f"Scaling factor needed: {10.82 / 1.40:.2f}x")
    
    print("\n🔧 LIKELY CAUSES")
    print("-" * 40)
    print("1. **Attention scaling**: Missing 1/sqrt(d_k) scaling")
    print("2. **RMS normalization**: Incorrect epsilon or scaling")
    print("3. **Weight orientation**: Transposed vs non-transposed")
    print("4. **RoPE scaling**: Incorrect frequency scaling")
    print("5. **FFN scaling**: Missing scaling factors")
    
    print("\n🎯 NEXT STEPS")
    print("-" * 40)
    print("1. Check attention scaling (1/sqrt(64) = 0.125)")
    print("2. Verify RMS normalization implementation")
    print("3. Check weight orientation in matrix multiplications")
    print("4. Verify RoPE frequency calculation")

if __name__ == "__main__":
    analyze_divergence()
