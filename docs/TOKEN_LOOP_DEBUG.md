# Token Loop Bug - Debugging Session

**Date**: 2025-10-06
**Status**: 🔴 CRITICAL BUG - Token looping prevents usable output

---

## 🎯 Current State

### What Works ✅
- Model loads successfully (15s for Q8 TinyLlama)
- Forward pass completes (~1.5s per token, 65ms per layer)
- Generation loop runs without crashes
- KV cache infrastructure functional
- Sampling mechanism executes

### The Bug ⚠️
**Symptom**: Model generates same token repeatedly in a loop

**Example Output** (Temperature=0.0, Greedy):
```
Prompt: "The meaning of life is"
Output: "The meaning of life isessen rect rect rect rect rect rect rect rect rect"
```

Token "rect" repeats infinitely (or until max_tokens).

---

## 🔍 Investigation Results

### What We Checked:

#### 1. KV Cache ✅ CORRECT
- **seq_pos tracking**: Increments by element count (256, 512, 768...)
- **This is correct**: seq_pos is element offset for slicing, not token count
- **kv_seq_len calculation**: `k.len() / (num_kv_heads * head_dim)` correctly gives token count
- **Math checks out**:
  - Token 0: seq_pos=256, kv_seq_len=1
  - Token 1: seq_pos=512, kv_seq_len=2
  - Token 2: seq_pos=768, kv_seq_len=3

#### 2. Causal Masking ✅ LOOKS CORRECT
```rust
let query_abs_pos = position + i;
if j > query_abs_pos {
    scores[j] = f32::NEG_INFINITY;
}
```
- Logic matches llama2.c pattern
- Unit test `test_attention_causal_masking` passes
- BUT: May have edge case we're missing

#### 3. Generation Loop ✅ CORRECT
- Follows llama2.c pattern exactly
- Processes prompt tokens correctly
- Samples from logits properly
- Loop termination logic sound

---

## 🐛 Possible Root Causes (Ranked by Likelihood)

### 1. **Softmax Numerical Issue** (HIGH)
**Location**: `transformer.rs:396-404`

**Code**:
```rust
let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
let mut exp_scores = vec![0.0; kv_seq_len];
let mut sum_exp = 0.0;

for j in 0..kv_seq_len {
    exp_scores[j] = (scores[j] - max_score).exp();
    sum_exp += exp_scores[j];
}
```

**Hypothesis**: If all scores are masked (`NEG_INFINITY`), max_score is `NEG_INFINITY`, leading to NaN:
- `(-INF) - (-INF) = NaN`
- `NaN.exp() = NaN`
- Attention becomes NaN, model outputs gibberish

**Test**: Add check for all-masked case

---

### 2. **RoPE Position Encoding Bug** (MEDIUM)
**Location**: `transformer.rs:208-209`

**Code**:
```rust
self.apply_rope(&mut q, position, seq_len, self.config.num_heads)?;
self.apply_rope(&mut k, position, seq_len, self.config.num_kv_heads)?;
```

**Hypothesis**: RoPE frequencies or application may be slightly off, causing:
- Similar queries to produce similar attention patterns
- Model "forgets" context and loops

**Test**: Compare RoPE implementation with llama2.c line-by-line

---

### 3. **Attention Score Computation** (MEDIUM)
**Location**: `transformer.rs:381`

**Code**:
```rust
let score = self.dot_product(q_vec, k_vec);
scores[j] = score * scale;
```

**Hypothesis**:
- Dot product implementation may have numerical instability
- Scale factor `1.0 / sqrt(head_dim)` may be applied incorrectly
- Accumulated errors cause attention to focus on one token

**Test**: Print attention scores for first few tokens, verify they're reasonable

---

### 4. **Sampling Determinism Issue** (LOW)
**Location**: `transformer.rs:1249-1254`

**Code**:
```rust
next = self.sample(&last_logits, temperature, top_p, top_k)?;
```

**Hypothesis**: With temperature=0.0 (greedy), sampling always picks same token
- This suggests logits are identical for every position
- Means the model's forward pass is producing same output regardless of context

**Test**: Print logits for positions 6, 7, 8 to see if they're identical

---

### 5. **Weight Transpose Bug** (LOW BUT CRITICAL IF TRUE)
**Location**: `transformer.rs:863-865, 927-941`

**Status**: Supposedly "fixed" per BUGS_FIXED.md

**Code**:
```rust
let wq_transposed = transpose_matrix(wq, self.config.hidden_size, self.config.hidden_size);
```

**Hypothesis**: Transpose may still be wrong for some matrices
- Or transpose function itself has a bug
- This would completely break the model

**Test**: Manually verify transpose_matrix function, check one weight matrix

---

## 🧪 Debugging Plan

### Phase 1: Add Diagnostic Logging (10 min)
```rust
// In compute_attention, after softmax:
eprintln!("Token {}, Head {}: attention weights = {:?}",
    position + i, h, &exp_scores[..5]);

// In generate loop:
eprintln!("Position {}: top 5 logits = {:?}",
    pos, top_k_logits(&last_logits, 5));
```

**Goal**: See if attention weights or logits are degenerate

---

### Phase 2: Test Softmax Edge Case (15 min)
```rust
// Before softmax, check for all-masked:
let valid_scores = scores.iter().filter(|&&s| s.is_finite()).count();
if valid_scores == 0 {
    eprintln!("WARNING: All scores masked at position {}!", position);
    // Fallback: uniform attention over valid positions
}
```

**Goal**: Fix potential NaN propagation

---

### Phase 3: Compare with llama2.c (30 min)
1. Find corresponding functions in `/home/puneet/llama2.c/run.c`
2. Compare:
   - Attention score calculation
   - RoPE frequency calculation
   - Softmax implementation
   - Sampling logic
3. Look for ANY differences, even tiny ones

**Goal**: Find the discrepancy

---

### Phase 4: Simplified Test Case (20 min)
Create minimal test:
```rust
#[test]
fn test_two_token_generation() {
    // Load model
    // Generate with prompt "Hello"
    // Check: first generated token should NOT be "Hello" again
    // Check: second generated token should be different from first
}
```

**Goal**: Reproducible test case for debugging

---

## 📊 Metrics to Track

When debugging, log these for each token:
1. **Position**: Absolute token position
2. **KV seq_len**: How many tokens in cache
3. **Attention entropy**: `- sum(p * log(p))` where p = attention weights
   - High entropy (>3) = attending to many tokens ✅
   - Low entropy (<1) = attending to one token ⚠️
   - Zero entropy = degenerate attention 🔴
4. **Logit range**: max - min of top 10 logits
   - Large range (>10) = confident predictions ✅
   - Small range (<1) = uncertain, flat distribution ⚠️
5. **Top token probability**: After softmax
   - Should vary across positions
   - If always same → model is "stuck"

---

## 🎯 Success Criteria

**Minimum**: No token loops for at least 50 tokens
**Good**: Output makes grammatical sense
**Ideal**: Output quality matches ollama with same model

---

## 🔗 Related Files

- `crates/wasm-chord-runtime/src/transformer.rs` - All core logic
- `docs/BUGS_FIXED.md` - Previous bug fixes (token loop was NOT fixed)
- `docs/debug_attention.md` - Attention masking analysis
- `/home/puneet/llama2.c/run.c` - Reference implementation

---

## 💡 Quick Wins to Try

1. **Test with temperature > 0**: Does randomness break the loop?
2. **Test with longer prompt**: Does more context help?
3. **Test different model**: Is it model-specific or code bug?
4. **Disable repetition penalty**: Is it making things worse?
5. **Check logits directly**: Print raw logits before sampling

---

## 🚀 Next Session Action Items

1. Add attention entropy logging
2. Test softmax all-masked edge case
3. Compare RoPE with llama2.c
4. If still stuck: Binary search - comment out layers until quality improves
5. Ultimate fallback: Copy working attention/RoPE from llama2.rs

**Time estimate**: 2-4 hours to fix, depending on root cause

---

**Remember**: The infrastructure works! This is purely a numerical/algorithmic bug. We're 90% there! 💪
