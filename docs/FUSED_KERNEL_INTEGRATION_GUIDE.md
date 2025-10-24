# Fused Kernel Integration Guide

**Date:** October 22, 2025  
**Status:** Foundation Complete (Steps 1-2 of 5)  
**Remaining:** Runtime Integration (Steps 3-5)

---

## ✅ Completed Work (Steps 1-2)

### Step 1: WeightFormat Enum ✅

**File:** `crates/wasm-chord-core/src/weight_format.rs` (NEW)

Created a unified weight storage enum:
```rust
pub enum WeightFormat {
    F32(Vec<f32>),
    Q4K(Vec<BlockQ4_K>),
    Q5K(Vec<BlockQ5_K>),
    Q6K(Vec<BlockQ6_K>),
    Q8K(Vec<BlockQ8_K>),
}
```

**Features:**
- `len()` - Get number of elements
- `memory_bytes()` - Get memory usage
- `compression_ratio()` - Compare to F32
- `format_name()` - Human-readable name

**Status:** ✅ Compiles, ready to use

### Step 2: TensorLoader Update ✅

**File:** `crates/wasm-chord-core/src/tensor_loader.rs`

Added new method:
```rust
pub fn load_weight_format<R: Read + Seek>(
    &mut self,
    name: &str,
    parser: &mut GGUFParser<R>,
) -> Result<WeightFormat>
```

**Behavior:**
- Q4_K/Q5_K/Q6_K/Q8_K → Keep as quantized blocks
- Q4_0/Q8_0 → Dequantize to F32 (no fused kernel yet)
- F32 → Return as-is

**Status:** ✅ Compiles, ready to use

---

## 🔄 Remaining Work (Steps 3-5)

### Step 3: Update Model Structures (~3-4 hours)

**Goal:** Change weight storage from `Vec<f32>` to `WeightFormat`

#### 3.1 Update AttentionWeights

**File:** `crates/wasm-chord-runtime/src/transformer/attention.rs`

**Current:**
```rust
pub struct AttentionWeights {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
}
```

**Change to:**
```rust
pub struct AttentionWeights {
    pub wq: WeightFormat,  // ← Change
    pub wk: WeightFormat,  // ← Change
    pub wv: WeightFormat,  // ← Change
    pub wo: WeightFormat,  // ← Change
}
```

**Update methods:**
- `AttentionWeights::new()` - Initialize with F32 empty vecs
- Any methods that access `.wq`, `.wk`, etc.

#### 3.2 Update FFNWeights

**File:** `crates/wasm-chord-runtime/src/transformer/ffn.rs`

**Current:**
```rust
pub struct FFNWeights {
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}
```

**Change to:**
```rust
pub struct FFNWeights {
    pub w_gate: WeightFormat,  // ← Change
    pub w_up: WeightFormat,    // ← Change
    pub w_down: WeightFormat,  // ← Change
}
```

#### 3.3 Update Model Loading

**File:** `crates/wasm-chord-runtime/src/transformer/model.rs`

**Current (lines 649-682):**
```rust
if let Ok(wq) = tensor_loader.load_tensor(&wq_name, parser) {
    layer.attention_weights.wq.copy_from_slice(wq);
}
```

**Change to:**
```rust
if let Ok(wq) = tensor_loader.load_weight_format(&wq_name, parser) {
    layer.attention_weights.wq = wq;  // ← Direct assignment
}
```

**Apply to all weight loads:**
- Attention: wq, wk, wv, wo
- FFN: w_gate, w_up, w_down  
- Embeddings (can stay F32 for now)
- LM head (can stay F32 for now)

---

### Step 4: Update Forward Pass (~2-3 hours)

**Goal:** Dispatch to fused kernels based on weight format

#### 4.1 Create Dispatch Helper

**File:** `crates/wasm-chord-runtime/src/transformer/mod.rs` (or new file)

```rust
use wasm_chord_core::WeightFormat;
use wasm_chord_cpu::{
    fused_dequant_matmul_q4k,
    fused_dequant_matmul_q5k,
    fused_dequant_matmul_q6k,
    fused_dequant_matmul_q8k,
    matmul_transposed,
};

/// Dispatch matmul to appropriate kernel based on weight format
pub fn dispatch_matmul(
    input: &[f32],        // [batch, k]
    weights: &WeightFormat,  // [n, k] (transposed)
    batch_size: usize,    // m
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    match weights {
        WeightFormat::F32(w) => {
            // Standard f32 matmul
            let mut output = vec![0.0f32; batch_size * n];
            matmul_transposed(input, w, &mut output, batch_size, k, n)?;
            Ok(output)
        }
        WeightFormat::Q4K(blocks) => {
            // Fused Q4_K kernel
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q4k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q5K(blocks) => {
            // Fused Q5_K kernel
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q5k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q6K(blocks) => {
            // Fused Q6_K kernel
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q6k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q8K(blocks) => {
            // Fused Q8_K kernel
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q8k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
    }
}
```

#### 4.2 Update Attention Forward

**File:** `crates/wasm-chord-runtime/src/transformer/attention.rs`

**Current (lines 84-93):**
```rust
let q = self.matmul(
    hidden_states,
    &weights.wq,  // ← This is now WeightFormat
    seq_len,
    hidden_size,
    hidden_size,
    true,
    #[cfg(feature = "webgpu")]
    gpu,
)?;
```

**Change to:**
```rust
let q = dispatch_matmul(
    hidden_states,
    &weights.wq,  // ← WeightFormat
    seq_len,      // batch
    hidden_size,  // k
    hidden_size,  // n
)?;
```

**Apply to:** Q, K, V, O projections

#### 4.3 Update FFN Forward

**File:** `crates/wasm-chord-runtime/src/transformer/ffn.rs`

Similar changes for:
- Gate projection
- Up projection
- Down projection

---

### Step 5: Test Integration (~1-2 hours)

#### 5.1 Compile Tests
```bash
cargo test -p wasm-chord-runtime --lib
```

**Fix any:**
- Type mismatches (Vec<f32> → WeightFormat)
- Method signature changes
- Test assertions

#### 5.2 Run Real Model Test
```bash
cargo run --release --example simple-generation
```

**Verify:**
- Model loads successfully
- Output is correct ("Paris is the capital of France")
- No crashes or errors

#### 5.3 Performance Benchmark
```bash
# Measure actual speedup
cargo bench --package wasm-chord-cpu --bench fused_kernels

# Or run generation with timing
time cargo run --release --example simple-generation
```

**Expected Results:**
- 2-4x faster inference
- 4-8x less memory usage
- Correct output quality

---

## 📋 Implementation Checklist

### Step 3: Model Structures
- [ ] Update `AttentionWeights` struct
- [ ] Update `AttentionWeights::new()`
- [ ] Update `FFNWeights` struct
- [ ] Update `FFNWeights::new()`
- [ ] Update model loading (attention weights)
- [ ] Update model loading (FFN weights)
- [ ] Fix compilation errors

### Step 4: Forward Pass
- [ ] Create `dispatch_matmul()` helper
- [ ] Update attention Q projection
- [ ] Update attention K projection
- [ ] Update attention V projection
- [ ] Update attention O projection
- [ ] Update FFN gate projection
- [ ] Update FFN up projection
- [ ] Update FFN down projection
- [ ] Fix compilation errors

### Step 5: Testing
- [ ] Fix test compilation
- [ ] Run unit tests
- [ ] Test with real model
- [ ] Verify correctness
- [ ] Benchmark performance
- [ ] Document results

---

## 🚀 Quick Start (Continue Integration)

```bash
cd /home/puneet/wasm-chord

# Step 3: Update model structures
# Edit: crates/wasm-chord-runtime/src/transformer/attention.rs
# Edit: crates/wasm-chord-runtime/src/transformer/ffn.rs  
# Edit: crates/wasm-chord-runtime/src/transformer/model.rs

# Step 4: Update forward pass
# Create: dispatch_matmul helper
# Update: All matmul calls

# Step 5: Test
cargo test -p wasm-chord-runtime --lib
cargo run --release --example simple-generation
```

---

## 📊 Expected Impact

### Before Integration
```
Load: GGUF → Dequantize to F32 → Store F32 (8x memory)
Inference: F32 matmul (slow, high bandwidth)
```

### After Integration
```
Load: GGUF → Keep quantized → Store blocks (1x memory)
Inference: Fused dequant+matmul (fast, low bandwidth)
```

### Performance Gains
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory | 8GB | 1GB | 8x less |
| Bandwidth | High | Low | 8x less |
| Speed | Baseline | 2-4x | 2-4x faster |

---

## 🛠️ Troubleshooting

### Common Issues

**Type Mismatches:**
```
error: mismatched types
expected `&[f32]`
found `&WeightFormat`
```
**Fix:** Use `dispatch_matmul()` instead of direct matmul

**Copy From Slice:**
```
error: no method `copy_from_slice` on `WeightFormat`
```
**Fix:** Direct assignment: `layer.wq = weight_format`

**Test Failures:**
```
error: cannot index into WeightFormat
```
**Fix:** Match on format or use helper methods

---

## 📝 Notes

1. **Embeddings & LM Head:** Can keep as F32 for now (used once per token)
2. **GPU Path:** Will need similar updates when GPU is available
3. **Backward Compatibility:** Old models will still work (F32 path)
4. **Memory Savings:** Immediate on load
5. **Speed Gains:** Immediate in forward pass

---

**Status:** Foundation complete, ready for runtime integration!

The hard infrastructure work is done. Steps 3-5 are mostly mechanical changes
to use the new `WeightFormat` type and `dispatch_matmul()` helper.

Estimated time to complete: **6-9 hours** spread across 1-2 days.

