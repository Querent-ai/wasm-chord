# Debug Attention Masking

## Current Implementation Analysis

### Causal Mask Logic (lines 287-292)
```rust
let query_abs_pos = position + i;
if j > query_abs_pos {
    scores[j] = f32::NEG_INFINITY;
}
```

### Test Cases

#### Prefill: tokens=[BOS, Hello] at position=0
- seq_len=2, position=0
- KV cache after append: contains positions 0,1
- kv_seq_len=2

Query i=0 (BOS at abs pos 0):
- query_abs_pos = 0 + 0 = 0
- j=0: 0 > 0? NO - attend to self ✓
- j=1: 1 > 0? YES - mask future token ✓

Query i=1 (Hello at abs pos 1):
- query_abs_pos = 0 + 1 = 1
- j=0: 0 > 1? NO - attend to BOS ✓
- j=1: 1 > 1? NO - attend to self ✓

#### Incremental: Generating position 2
- seq_len=1, position=2
- KV cache: contains positions 0,1,2 (just appended position 2)
- kv_seq_len=3

Query i=0 (new token at abs pos 2):
- query_abs_pos = 2 + 0 = 2
- j=0: 0 > 2? NO - attend ✓
- j=1: 1 > 2? NO - attend ✓
- j=2: 2 > 2? NO - attend to self ✓

**This all looks correct!**

## Hypothesis

The masking is correct, so the bug must be elsewhere. Possible culprits:
1. Weight loading issue
2. RoPE bug
3. FFN bug
4. Normalization bug
5. Matrix multiplication bug

Next: Compare against a minimal reference implementation
