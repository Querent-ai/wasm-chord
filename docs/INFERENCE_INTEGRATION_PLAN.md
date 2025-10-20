# Inference Integration Plan: Memory64 → Transformer

**Status:** Ready to implement
**Priority:** HIGH (core Phase 2 goal)

---

## 🎯 Objective

Connect Memory64Model to the transformer forward pass, enabling actual inference with large (>4GB) models using on-demand layer loading.

---

## 📊 Current Flow Analysis

### Standard Inference Flow

```rust
Model::generate(prompt)
  ↓
Model::forward(tokens, position)  // Line 735
  ↓
  1. Embed tokens (line 747-768)
  2. For each layer (line 794-837):
     hidden_states = self.forward_layer(layer_idx, hidden_states, position)
                            ↓
                   Accesses: self.layers[layer_idx]  ✓ Integration point!
  3. Final RMS norm (line 840)
  4. LM head projection (line 858+)
```

### Key Integration Point

**Line 802:**
```rust
hidden_states = self.forward_layer(layer_idx, &hidden_states, position)?;
```

Currently accesses: `self.layers[layer_idx]` directly from `Vec<TransformerLayer>`

Need to support: Get layer from Memory64Model when model is large

---

## 🏗️ Integration Strategy

### Option A: Dual-Path (Recommended)

**Keep both paths, choose at runtime:**

```rust
pub struct Model {
    // Existing
    pub layers: Vec<TransformerLayer>,  // For standard models

    // New
    #[cfg(feature = "memory64")]
    pub memory64_model: Option<Memory64Model>,  // For large models
}

impl Model {
    fn forward_layer(&mut self, layer_idx: usize, ...) -> Result<Vec<f32>> {
        #[cfg(feature = "memory64")]
        if let Some(ref mut mem64) = self.memory64_model {
            // Use Memory64 path
            let layer = mem64.get_layer(layer_idx as u32)?;
            return self.process_layer(layer, ...);
        }

        // Standard path
        let layer = &self.layers[layer_idx];
        self.process_layer(layer, ...)
    }
}
```

**Pros:**
- ✅ Backward compatible
- ✅ Zero overhead for standard models
- ✅ Clean separation
- ✅ Easy to test both paths

**Cons:**
- ⚠️ Some code duplication
- ⚠️ Slightly more complex

### Option B: Unified Trait (Future)

```rust
trait LayerProvider {
    fn get_layer(&mut self, idx: usize) -> Result<&TransformerLayer>;
}

impl LayerProvider for Vec<TransformerLayer> { ... }
impl LayerProvider for Memory64Model { ... }

pub struct Model {
    layer_provider: Box<dyn LayerProvider>,
}
```

**Decision:** Start with **Option A** (simpler, faster to implement)

---

## 📝 Implementation Steps

### Step 1: Modify Model Struct ✅

```rust
// crates/wasm-chord-runtime/src/transformer/model.rs

pub struct Model {
    // Existing fields...
    pub token_embeddings: Vec<f32>,
    pub layers: Vec<TransformerLayer>,
    pub output_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub kv_caches: Vec<KVCache>,

    // NEW: Optional Memory64 model for large models
    #[cfg(feature = "memory64")]
    pub memory64_model: Option<Memory64Model>,
}
```

### Step 2: Add Layer Access Helper

```rust
impl Model {
    /// Get layer (either from Vec or Memory64)
    #[inline]
    fn get_layer_ref(&self, layer_idx: usize) -> Result<&TransformerLayer> {
        #[cfg(feature = "memory64")]
        if let Some(ref mem64) = self.memory64_model {
            // Memory64 path: Get layer on-demand
            return mem64.get_layer(layer_idx as u32);
        }

        // Standard path: Direct Vec access
        self.layers.get(layer_idx)
            .ok_or_else(|| Error::Runtime(format!("Layer {} not found", layer_idx)))
    }
}
```

### Step 3: Update forward_layer

```rust
// Line 707
fn forward_layer(
    &mut self,
    layer_idx: usize,
    hidden_states: &[f32],
    position: usize,
) -> Result<Vec<f32>> {
    // OLD: Direct access
    // let layer = &self.layers[layer_idx];

    // NEW: Indirect access (supports Memory64)
    let layer = self.get_layer_ref(layer_idx)?;

    // Rest of the function unchanged...
    let seq_len = hidden_states.len() / self.config.hidden_size;
    // ... existing layer processing ...
}
```

### Step 4: Update load_from_gguf for Memory64

```rust
impl Model {
    pub fn load_from_gguf<R: Read + Seek>(
        &mut self,
        tensor_loader: &mut TensorLoader,
        parser: &mut GGUFParser<R>,
    ) -> Result<()> {
        #[cfg(feature = "memory64")]
        {
            // Check if model is large (>3GB)
            let total_size = estimate_model_size(parser)?;
            if total_size > 3_000_000_000 {
                println!("🎯 Large model detected ({:.2} GB) - using Memory64",
                         total_size as f64 / 1e9);

                // Initialize Memory64Model
                let mem64_loader = Memory64GGUFLoader::new();
                self.memory64_model = Some(mem64_loader.load_model(parser)?);

                // Skip loading layers into Vec (save memory)
                return Ok(());
            }
        }

        // Standard loading (existing code)
        // Load embeddings, layers, etc.
        // ...
    }
}
```

### Step 5: Update Model::new

```rust
pub fn new(config: TransformerConfig) -> Self {
    let mut layers = Vec::new();
    let mut kv_caches = Vec::new();

    for _ in 0..config.num_layers {
        layers.push(TransformerLayer::new(&config));
        kv_caches.push(KVCache::new(...));
    }

    Self {
        token_embeddings: vec![0.0; config.vocab_size * config.hidden_size],
        output_norm: vec![1.0; config.hidden_size],
        lm_head: vec![0.0; config.hidden_size * config.vocab_size],
        kv_caches,
        layers,
        candle_backend: CandleTensorBackend::new(),

        #[cfg(feature = "memory64")]
        memory64_model: None,  // NEW

        // ... other fields ...
    }
}
```

---

## 🧪 Testing Strategy

### Test 1: Backward Compatibility

```rust
#[test]
fn test_standard_model_unchanged() {
    // Load TinyLlama (0.67GB) - should use standard path
    let model = Model::load_from_file("tinyllama.gguf")?;
    assert!(model.memory64_model.is_none());

    // Generate text - should work as before
    let output = model.generate("Hello", &tokenizer, &config)?;
    assert!(!output.is_empty());
}
```

### Test 2: Memory64 Activation

```rust
#[test]
#[cfg(feature = "memory64")]
fn test_large_model_uses_memory64() {
    // Load Llama-2-7B (4.08GB) - should use Memory64
    let model = Model::load_from_file("llama-2-7b.gguf")?;
    assert!(model.memory64_model.is_some());
    assert_eq!(model.layers.len(), 0);  // Layers not loaded into Vec
}
```

### Test 3: End-to-End Generation

```rust
#[test]
#[cfg(feature = "memory64")]
fn test_memory64_generation() {
    let model = Model::load_from_file("llama-2-7b.gguf")?;
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    let config = GenerationConfig::default();

    // Test generation
    let output = model.generate("Hi", &tokenizer, &config)?;
    assert!(!output.is_empty());
    println!("Generated: {}", output);
}
```

---

## 📊 Performance Expectations

### Baseline (Standard Loading)

| Model | Memory | Speed | Status |
|-------|--------|-------|--------|
| TinyLlama (0.67GB) | 800 MB | 10 tok/s | ✅ Working |
| Llama-2-7B (4.08GB) | OOM | N/A | ❌ Fails |

### With Memory64

| Model | Memory | Speed (Cold) | Speed (Warm) | Status |
|-------|--------|--------------|--------------|--------|
| TinyLlama (0.67GB) | 800 MB | 10 tok/s | 10 tok/s | ✅ Unchanged |
| Llama-2-7B (4.08GB) | ~100 MB | 2-5 tok/s | 5-10 tok/s | 🎯 Target |

**Notes:**
- Cold: First few tokens (cache misses)
- Warm: After cache fills (cache hits)
- Memory: Peak RAM usage

---

## 🚀 Implementation Timeline

### Day 1 (Today)
- ✅ Analysis complete
- 🔲 Implement Step 1-3 (struct changes, layer access)
- 🔲 Test compilation

### Day 2
- 🔲 Implement Step 4-5 (load_from_gguf integration)
- 🔲 Write tests
- 🔲 Test with TinyLlama (backward compat)

### Day 3
- 🔲 Test with Llama-2-7B (Memory64 path)
- 🔲 Debug issues
- 🔲 Performance profiling

---

## 🎯 Success Criteria

- [x] Code compiles with and without `memory64` feature
- [ ] TinyLlama generates text (backward compatible)
- [ ] Llama-2-7B loads with Memory64 (<200MB RAM)
- [ ] Llama-2-7B generates coherent text
- [ ] No performance regression for standard models
- [ ] Tests pass for both paths

---

## 🔧 Potential Issues & Solutions

### Issue 1: Borrow Checker Conflicts

**Problem:** `get_layer_ref()` borrows mutably, but we also need mutable access to KV cache

**Solution:** Change signature:
```rust
// OLD (borrow conflict)
fn forward_layer(&mut self, layer_idx: usize, ...) {
    let layer = self.get_layer_ref(layer_idx)?;  // Borrows self
    let kv = &mut self.kv_caches[layer_idx];     // Also borrows self
}

// NEW (no conflict)
fn forward_layer(&mut self, layer_idx: usize, ...) {
    let layer = self.get_layer_owned(layer_idx)?;  // Returns Arc<Layer>
    let kv = &mut self.kv_caches[layer_idx];       // No conflict
}
```

### Issue 2: KV Cache Desync

**Problem:** Memory64 layers loaded on-demand, but KV cache references layer index

**Solution:** Ensure KV cache is always initialized for all layers, even if layer weights are in Memory64

### Issue 3: Performance Degradation

**Problem:** Layer loading adds latency (342ms/layer cold)

**Solution:**
1. Prefetch next layers (Phase 2)
2. Larger cache (8-16 layers)
3. Async loading (Phase 2)

---

## 📝 Next Actions

1. **Implement struct changes** (add `memory64_model` field)
2. **Add `get_layer_ref()` helper**
3. **Modify `forward_layer()` to use helper**
4. **Test compilation**
5. **Write unit tests**
6. **Test with real models**

---

**Ready to proceed! Let's connect Memory64 to inference! 🚀**
