# Debugging Log - Token Generation Issue

## Problem Statement
Model generates poor quality output compared to ollama/llama.cpp with same GGUF file.
- With greedy sampling (temp=0): Loops on token 19762 ("automatisch")
- With sampling: Generates gibberish but varying tokens
- Token 19762 consistently has highest logit (~5.6-5.7) across ALL positions

## What We've Verified is CORRECT ✅

1. **GGUF Weight Loading**
   - No transpose needed when loading from GGUF
   - Weights stored as `[in_features, out_features]`
   - Matches llama.cpp and Candle implementations
   - Verified with diagnostic logging

2. **Q8_0 Dequantization**
   - Formula: `quant * scale` where scale is f16
   - Matches Candle's implementation exactly
   - No NaN/Inf values in loaded weights

3. **RoPE (Rotary Position Embedding)**
   - Tested BOTH modes:
     - Interleaved pairs: `[0,1], [2,3], ...`
     - Contiguous pairs: `[0, d/2], [1, d/2+1], ...`
   - Currently using contiguous mode (matches Candle)
   - Both modes produce same token loop issue

4. **Embeddings**
   - Values look normal: mean~0.0001, range~[-0.03, 0.04]
   - Properly indexed by token ID

5. **LM Head Weights**
   - Token 19762 column has no unusual bias
   - abs_mean=0.0086 (similar to other tokens ~0.008-0.009)
   - Weight distribution looks normal

6. **Forward Pass Structure**
   - Correct order: Embedding → Layers → Final Norm → LM Head
   - Residual connections properly added
   - Matches reference implementations

## Debug Output Analysis

### Logits Across Positions (with DEBUG_FORWARD=1)

**Position 0 (BOS token):**
```
Embeddings: mean=0.000123, range=[-0.006, 0.006]
After final norm: mean=-0.065, range=[-5.148, 4.171]
Logits: mean=-0.007, range=[-3.831, 4.554]
Top 5: [(19762, "5.62"), (15327, "5.34"), (3676, "5.08"), ...]
```

**Position 1 (token "The"):**
```
Embeddings: mean=0.000046, range=[-0.023, 0.021]
After final norm: mean=-0.069, range=[-5.144, 3.997]
Logits: mean=-0.007, range=[-3.957, 4.521]
Top 5: [(19762, "5.63"), (15327, "5.01"), (4608, "4.92"), ...]
```

**Position 2 (token "meaning"):**
```
Embeddings: mean=0.001115, range=[-0.033, 0.035]
After final norm: mean=-0.068, range=[-5.079, 4.139]
Logits: mean=-0.004, range=[-3.965, 4.556]
Top 5: [(19762, "5.61"), (21434, "5.03"), (14656, "4.92"), ...]
```

### Key Observations

1. **Logit ranges barely change**: All positions show range ~[-4, 5]
2. **Token 19762 dominates**: Always ~5.6-5.7, next best ~5.0-5.3
3. **Final norm values**: Consistent mean~-0.07, range~[-5, 4]
4. **Embeddings vary properly**: Different tokens have different embedding stats

## Possible Root Causes (Ordered by Likelihood)

### 1. Hidden States Not Varying Between Positions ⚠️
The constant logit ranges suggest hidden states aren't incorporating positional/contextual information properly.

**Could be:**
- KV cache not being updated correctly
- KV cache being read from wrong positions
- Position parameter not being used properly in attention

### 2. Attention Mechanism Issue
- Q/K/V projections might have dimension mismatch
- Attention scores not being computed correctly
- Values not being aggregated properly

### 3. RMS Normalization Bug
- Scale factors might be wrong
- Epsilon value incorrect
- Normalization applied incorrectly

### 4. FFN (Feed-Forward Network) Issue
- Gate/Up/Down projections might have bugs
- SiLU activation not applied correctly
- Intermediate size mismatch

### 5. Subtle Matmul Bug
- Despite verification, could have off-by-one or stride issue
- Dimension interpretation might be subtly wrong

## References Compared

1. **llama.cpp** (`/home/puneet/llama.cpp`)
   - Uses `ggml_mul_mat` which transposes second operand internally
   - We verified our approach matches when accounting for this

2. **Candle** (`/home/puneet/candle`)
   - RoPE uses contiguous mode (matches our current impl)
   - Q8_0 dequant matches exactly
   - Weight loading matches (no transpose)

3. **llama2.c** (`/home/puneet/llama2.c`)
   - Uses custom checkpoint format (not GGUF)
   - Matmul convention: `W[out, in] @ x[in] = out[out]`
   - Helped verify our understanding

4. **llama-rs** (`/home/puneet/llama-rs`)
   - NEW - not yet examined

## Next Steps

### Immediate Actions
1. ✅ Add layer-wise debugging to see where hidden states stop varying
2. ⏳ Compare KV cache management with reference implementations
3. ⏳ Check llama-rs implementation for clues
4. ⏳ Add numerical checks: compute attention entropy, check if Q@K^T produces reasonable scores

### Tests to Run
1. **Single Layer Test**: Run just layer 0 and check output
2. **No KV Cache Test**: Disable KV caching to see if that's the issue
3. **Position=0 Only**: Generate with all positions at 0 to isolate position encoding
4. **Compare Intermediate Values**: Match our hidden states against Candle's at each step

### Debug Commands
```bash
# Enhanced debug output
env DEBUG_FORWARD=1 ./target/release/simple-generation 2>&1 | head -80

# Check specific tokens
env DEBUG_LOGITS=1 ./target/release/simple-generation 2>&1 | grep "19762"

# Profile layers
env PROFILE=1 ./target/release/simple-generation 2>&1
```

## Files Modified

- `crates/wasm-chord-runtime/src/transformer.rs`: Added extensive debug logging
- `examples/simple-generation/main.rs`: Adjusted generation config for testing
- `examples/check-lm-head/`: New tool to check lm_head weight bias
- Various RoPE and weight loading experiments (reverted after testing)

## Conclusion

The bug is subtle but systematic. The model IS running (generates tokens), but hidden states aren't varying enough between positions, causing poor output quality. Most likely culprit is in how KV cache is managed or how positions are used in attention. The fact that ollama works perfectly with the same file proves the GGUF file is correct - the bug is definitely in our implementation.

**Confidence**: Will find and fix this bug. Just need to trace through one forward pass step-by-step comparing with a working implementation.
