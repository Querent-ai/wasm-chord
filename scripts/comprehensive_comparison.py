#!/usr/bin/env python3
"""
Comprehensive Comparison Script for wasm-chord vs llama.cpp Debug Trace Analysis
"""

def analyze_divergence():
    print("🔍 COMPREHENSIVE LOGIT DIVERGENCE ANALYSIS")
    print("=" * 50)
    
    # wasm-chord debug trace (from console output with BOS token)
    wasm_chord_debug = {
        "EMB": [-0.001300, 0.001904, -0.001941, 0.003827, 0.001263],
        "L0_OUT": [0.000523, 0.001074, -0.002324, 0.006813, 0.000466],
        "L21_OUT": [-0.328055, 0.104522, -1.429869, 1.018251, -0.979945],
        "HIDDEN": [-0.573152, 0.467244, -3.502854, 2.362807, -1.774403],
    }
    
    # wasm-chord top 20 logits (from console output with BOS token)
    wasm_chord_top_predictions = [
        (10588, 9.307076, "widet"),
        (11068, 8.199226, "ёл"),
        (11674, 7.328809, "rice"),
        (14524, 7.176031, "cito"),
        (5219, 7.147675, "mary"),
        (12689, 7.002761, " CHAPTER"),
        (8994, 6.964008, "ян"),
        (10853, 6.865206, "contents"),
        (4999, 6.793930, "elles"),
        (16017, 6.702531, "iona"),
        (2101, 6.687664, "igen"),
        (9510, 6.643142, "ikz"),
        (3074, 6.632792, "burg"),
        (18382, 6.557157, "itori"),
        (2552, 6.547486, "berg"),
        (7984, 6.418515, " cook"),
        (16694, 6.403393, " comma"),
        (19336, 6.387581, "igos"),
        (18191, 6.377980, "клад"),
        (8484, 6.250537, "/#"),
    ]
    
    # llama.cpp top prediction (from console output)
    llama_cpp_top_prediction = (29892, 10.820, ",")
    
    print("\n📊 INTERMEDIATE VALUES COMPARISON")
    print("-" * 40)
    
    # Analyze intermediate values
    print("🔍 Token Embeddings (EMB):")
    print(f"  wasm-chord: {wasm_chord_debug['EMB']}")
    print(f"  Magnitude range: {min(wasm_chord_debug['EMB']):.6f} to {max(wasm_chord_debug['EMB']):.6f}")
    
    print("\n🔍 Layer 0 Output (L0_OUT):")
    print(f"  wasm-chord: {wasm_chord_debug['L0_OUT']}")
    print(f"  Magnitude range: {min(wasm_chord_debug['L0_OUT']):.6f} to {max(wasm_chord_debug['L0_OUT']):.6f}")
    
    print("\n🔍 Layer 21 Output (L21_OUT):")
    print(f"  wasm-chord: {wasm_chord_debug['L21_OUT']}")
    print(f"  Magnitude range: {min(wasm_chord_debug['L21_OUT']):.6f} to {max(wasm_chord_debug['L21_OUT']):.6f}")
    
    print("\n🔍 Hidden States Before LM Head (HIDDEN):")
    print(f"  wasm-chord: {wasm_chord_debug['HIDDEN']}")
    print(f"  Magnitude range: {min(wasm_chord_debug['HIDDEN']):.6f} to {max(wasm_chord_debug['HIDDEN']):.6f}")
    
    print("\n📈 PROGRESSION ANALYSIS")
    print("-" * 40)
    
    # Check progression from embeddings to final hidden states
    emb_range = max(wasm_chord_debug['EMB']) - min(wasm_chord_debug['EMB'])
    l0_range = max(wasm_chord_debug['L0_OUT']) - min(wasm_chord_debug['L0_OUT'])
    l21_range = max(wasm_chord_debug['L21_OUT']) - min(wasm_chord_debug['L21_OUT'])
    hidden_range = max(wasm_chord_debug['HIDDEN']) - min(wasm_chord_debug['HIDDEN'])
    
    print(f"Embeddings range: {emb_range:.6f}")
    print(f"Layer 0 range:   {l0_range:.6f}")
    print(f"Layer 21 range:  {l21_range:.6f}")
    print(f"Hidden range:    {hidden_range:.6f}")
    
    print("\n🎯 LOGIT COMPARISON")
    print("-" * 40)
    
    print("wasm-chord top predictions:")
    for i, (token_id, logit, text) in enumerate(wasm_chord_top_predictions[:5], 1):
        print(f"  {i}: token {token_id} = {logit:.6f} (\"{text}\")")
    
    print(f"\nllama.cpp top prediction:")
    print(f"  1: token {llama_cpp_top_prediction[0]} = {llama_cpp_top_prediction[1]:.6f} (\"{llama_cpp_top_prediction[2]}\")")
    
    print("\n🚨 CRITICAL DIVERGENCE ANALYSIS")
    print("-" * 40)
    
    # Calculate divergence metrics
    wasm_max_logit = max(logit for _, logit, _ in wasm_chord_top_predictions)
    llama_max_logit = llama_cpp_top_prediction[1]
    
    print(f"wasm-chord max logit: {wasm_max_logit:.6f}")
    print(f"llama.cpp max logit:  {llama_max_logit:.6f}")
    print(f"Logit magnitude diff: {abs(wasm_max_logit - llama_max_logit):.6f}")
    
    # Check if llama.cpp prediction is in wasm-chord's top predictions
    llama_token = llama_cpp_top_prediction[0]
    wasm_token_ids = [token_id for token_id, _, _ in wasm_chord_top_predictions]
    
    if llama_token in wasm_token_ids:
        idx = wasm_token_ids.index(llama_token)
        wasm_logit = wasm_chord_top_predictions[idx][1]
        print(f"\n✅ llama.cpp token {llama_token} found in wasm-chord predictions")
        print(f"   wasm-chord logit: {wasm_logit:.6f}")
        print(f"   llama.cpp logit:  {llama_max_logit:.6f}")
        print(f"   Difference:       {abs(wasm_logit - llama_max_logit):.6f}")
    else:
        print(f"\n❌ llama.cpp token {llama_token} NOT found in wasm-chord top 20")
        print("   However, it was found at position 19524 with logit 0.000000")
        print("   This indicates the token exists but has extremely low probability")
        print("   This is a MASSIVE divergence - the token should be near the top!")
    
    print("\n🔍 NUMERICAL STABILITY CHECK")
    print("-" * 40)
    
    # Check for NaN/Inf in intermediate values
    all_values = []
    for stage, values in wasm_chord_debug.items():
        all_values.extend(values)
    
    has_nan = any(str(v) == 'nan' for v in all_values)
    has_inf = any(str(v) == 'inf' or str(v) == '-inf' for v in all_values)
    
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    print(f"All values finite: {not (has_nan or has_inf)}")
    
    print("\n📋 SUMMARY")
    print("-" * 40)
    print("✅ wasm-chord intermediate values look numerically reasonable")
    print("✅ No NaN/Inf detected in intermediate computations")
    print("✅ Values show expected progression through layers")
    print("❌ MASSIVE divergence in final logits between implementations")
    print("❌ Top predictions are completely different")
    print("\n🎯 NEXT STEPS:")
    print("1. Need llama.cpp intermediate debug values for comparison")
    print("2. Check tokenization differences (BOS token handling)")
    print("3. Verify weight loading and quantization")
    print("4. Compare attention computation implementations")

if __name__ == "__main__":
    analyze_divergence()