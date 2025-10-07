# Next Session Preparation

## Current Status
✅ Bug isolated to: **KV cache or position handling in attention**
✅ All weight loading, RoPE, dequantization verified correct
✅ Debug logging in place throughout forward pass
✅ Three reference implementations available

## You Have Everything Needed! ✅

### Reference Implementations (Already Cloned)
1. **`/home/puneet/llama.cpp`** - C++ reference (most mature)
2. **`/home/puneet/llama-rs`** - Rust implementation (closest to our code)
3. **`/home/puneet/candle`** - Rust ML framework (already compared)
4. **`/home/puneet/llama2.c`** - Minimal C implementation (already compared)

**Recommendation**: Focus on **llama-rs** first - it's Rust and likely has the clearest KV cache implementation to compare.

## No Additional Clones Needed

Everything required is already in place:
- ✅ Reference implementations
- ✅ Debug tooling in wasm-chord
- ✅ GGUF model file (`models/tinyllama-q8.gguf`)
- ✅ ollama for baseline comparison

## Quick Start Commands for Next Session

### 1. Review the Bug Analysis
```bash
cd /home/puneet/wasm-chord
cat DEBUG_LOG.md
```

### 2. Check llama-rs KV Cache Implementation
```bash
cd /home/puneet/llama-rs
find . -name "*.rs" | xargs grep -l "kv_cache\|KVCache" | head -5
```

### 3. Run Our Debug Build
```bash
cd /home/puneet/wasm-chord
env DEBUG_FORWARD=1 ./target/release/simple-generation 2>&1 | head -80
```

### 4. Compare with Ollama (Ground Truth)
```bash
echo "The meaning of life is" | ollama run tinyllama --verbose
```

## Specific Investigation Plan

### Priority 1: KV Cache Management 🎯

**Check these in our code:**
```bash
cd /home/puneet/wasm-chord
grep -n "kv_cache.append\|kv_cache.seq_pos" crates/wasm-chord-runtime/src/transformer.rs
```

**Compare with llama-rs:**
```bash
cd /home/puneet/llama-rs
# Find their KV cache implementation
find . -type f -name "*.rs" -exec grep -l "cache.*position\|kv.*cache" {} \;
```

**Key questions to answer:**
1. Are we appending to KV cache at the RIGHT position?
2. Are we reading from KV cache with the RIGHT slice?
3. Is `position` parameter being used correctly in attention?
4. Is KV cache growing properly as we generate tokens?

### Priority 2: Attention Position Usage

**In our code, check:**
```rust
// crates/wasm-chord-runtime/src/transformer.rs
// Look at compute_attention function - line ~300-450
// Verify:
// 1. Position used correctly for causal masking
// 2. KV cache slice is correct: kv_cache.keys[..kv_cache.seq_pos]
// 3. Attention computed over ALL cached keys/values
```

### Priority 3: Position Encoding

**Verify position parameter flow:**
```bash
cd /home/puneet/wasm-chord
grep -n "position" crates/wasm-chord-runtime/src/transformer.rs | head -20
```

**Check if we're:**
- ✅ Passing correct position to each layer
- ✅ Using position in RoPE correctly
- ✅ Incrementing position properly in generation loop

## Debug Helpers Already Built

### 1. Layer Output Inspector
```bash
env DEBUG_FORWARD=1 ./target/release/simple-generation 2>&1 | grep "After layer"
```

### 2. KV Cache State Inspector
```bash
env DEBUG_KV=1 ./target/release/simple-generation 2>&1 | grep "kv_cache.seq_pos"
```

### 3. Attention Weights Inspector
```bash
env DEBUG_ATTN_WEIGHTS=1 ./target/release/simple-generation 2>&1 | grep "Attention weights"
```

### 4. Logits Inspector
```bash
env DEBUG_FORWARD=1 ./target/release/simple-generation 2>&1 | grep "Logits:"
```

## Test Strategy

### Test 1: Disable KV Cache (Nuclear Option)
Temporarily modify code to NOT use KV cache - recompute everything each time.
If this fixes output → KV cache is the bug.

### Test 2: Single Position Test
Force all positions to 0 to see if position handling is the issue.

### Test 3: Layer 0 Only Test
Run only the first transformer layer to isolate where things break.

### Test 4: Compare Attention Scores
Add logging to print actual attention scores and compare with expected values.

## Files to Focus On

### Primary Investigation
1. **`crates/wasm-chord-runtime/src/transformer.rs`**
   - Lines 200-450: `compute_attention` function
   - Lines 160-250: `apply_rope` function
   - Lines 1200-1280: Generation loop

2. **Compare with llama-rs:**
   - Find their attention implementation
   - Find their KV cache management
   - Find their position handling

### Supporting Files
- `DEBUG_LOG.md` - All our findings
- `examples/simple-generation/main.rs` - Test harness
- `crates/wasm-chord-runtime/src/transformer.rs:70-110` - KVCache struct

## Expected Outcome

We should find ONE of these issues:

1. **KV Cache Position Bug**:
   - Writing to wrong offset
   - Reading wrong slice
   - Not incrementing seq_pos correctly

2. **Attention Position Bug**:
   - Position parameter not used in causal masking
   - Wrong position passed to RoPE
   - Position not properly offsetting KV cache reads

3. **Causal Masking Bug**:
   - Not masking future positions
   - Masking applied incorrectly

## Success Criteria

When we fix the bug, we should see:
- ✅ Top logit changes significantly between positions
- ✅ Logit range varies (not constant [-4, 5])
- ✅ Generated text is coherent like ollama's output
- ✅ Different tokens become top prediction at different positions

## Quick Sanity Checks

### Before starting:
```bash
# Verify ollama still works
echo "The meaning of life is" | ollama run tinyllama

# Verify our build is current
cd /home/puneet/wasm-chord && cargo build --release --bin simple-generation

# Verify debug output works
env DEBUG_FORWARD=1 timeout 5 ./target/release/simple-generation 2>&1 | head -20
```

## Notes

- The bug is DEFINITELY in our code (ollama works perfectly)
- The bug is SUBTLE (most things are correct)
- The bug is SYSTEMATIC (affects all positions consistently)
- The bug is likely a SINGLE LINE or small logic error

**Confidence Level**: Very high. With focused KV cache comparison, we should find and fix this in the next session.

## One More Thing to Try (Quick Win Possibility)

Before deep diving into llama-rs comparison, try this quick fix:

```rust
// In compute_attention, verify we're using the RIGHT slice:
// WRONG: &kv_cache.keys[..position]
// RIGHT: &kv_cache.keys[..kv_cache.seq_pos]

// Check if we have this bug!
```

This is a common mistake that would cause exactly the symptoms we see.
