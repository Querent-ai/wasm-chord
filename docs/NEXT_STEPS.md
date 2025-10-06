# Next Steps - Action Plan

## Current Situation
- ✅ Q4_K_M model downloaded (638MB, proper format)
- ✅ All infrastructure working (parsing, loading, inference pipeline)
- ❌ Current Q4_0 model uses non-standard 16-byte blocks (no scales)
- ❌ Need to implement Q4_K dequantization to use new model

## Recommended Path: Implement Q4_K Support

### Step 1: Implement Q4_K Dequantization (~1-2 hours)

**Add Q4_K block structure** (`crates/wasm-chord-core/src/quant.rs`):
```rust
/// Q4_K block: 256 4-bit values in super-block
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_K {
    pub d: u16,              // f16 super-block scale
    pub dmin: u16,           // f16 super-block min
    pub scales: [u8; 12],    // Quantized scales for 8 sub-blocks
    pub qs: [u8; QK_K / 2],  // 128 bytes of 4-bit quants
}

pub fn dequantize_q4_k(block: &BlockQ4_K, output: &mut [f32]) -> Result<()> {
    // Implementation based on ggml spec
    // ...
}
```

**Wire up to tensor loader** (`crates/wasm-chord-core/src/tensor_loader.rs`):
```rust
DataType::Q4_K => {
    self.dequantize_q4_k(&raw_data, metadata.desc.element_count())?
}
```

**Add test**:
```rust
#[test]
fn test_q4_k_dequant() {
    // Verify Q4_K dequantization works
}
```

### Step 2: Test with Q4_K_M Model (~30 min)

1. Run weight loading test:
   ```bash
   cargo test --release test_load_real_weights -- --ignored --nocapture
   ```

2. Verify:
   - No NaN/inf in weights
   - Finite logits after forward pass
   - Reasonable output distribution

3. Check inference:
   ```bash
   cargo test --release test_inference_with_real_weights -- --ignored --nocapture
   ```

### Step 3: Validate & Ship v0.1.0 (~1 hour)

1. **Run full test suite**:
   ```bash
   cargo test --release --all
   make check
   ```

2. **Benchmark performance**:
   ```bash
   cargo bench
   ```

3. **Create release**:
   - Tag version: `git tag v0.1.0`
   - Update CHANGELOG.md
   - Create GitHub release

4. **Publish NPM package**:
   ```bash
   cd npm-package
   npm publish
   ```

## Alternative: Quick F16 Test

If Q4_K takes too long, download F16 model for immediate validation:

```bash
cd models
curl -L "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.F16.gguf" -o tinyllama-f16.gguf
```

F16 requires no quantization logic - just direct f16→f32 conversion (already implemented).

## After Working Inference

### Phase 1 Completion (2-3 hours)
1. **Tokenizer integration**
   - Load vocab from GGUF metadata
   - BPE encode/decode
   - Test roundtrip

2. **End-to-end text generation**
   - "Hello, world!" → tokens → inference → text
   - Validate coherent output
   - Benchmark tokens/second

3. **Documentation**
   - Update README with usage examples
   - Add architecture docs
   - Performance numbers

### Phase 2 Preview (2-3 weeks)
4. **WebGPU backend**
   - GPU matmul kernel
   - GPU dequantization
   - 5-10x speedup target

5. **Model caching**
   - IndexedDB (browser)
   - FS cache (Node)
   - Faster subsequent loads

6. **Advanced features**
   - Streaming generation
   - Batch inference
   - Multi-model support

## Time Estimates

| Task | Time | Status |
|------|------|--------|
| Q4_K implementation | 1-2 hours | 📋 TODO |
| Test & validate | 30 min | 📋 TODO |
| Tokenizer | 1-2 hours | 📋 TODO |
| E2E generation | 30 min | 📋 TODO |
| Documentation | 1 hour | 📋 TODO |
| **Total to v0.1.0** | **4-6 hours** | 🎯 Ready |

## Success Criteria

Phase 1 complete when:
- ✅ Real model weights load without errors
- ✅ Forward pass produces finite logits
- ✅ Sampling generates valid tokens
- ✅ Tokenizer works (encode/decode)
- ✅ E2E: text input → text output
- ✅ All tests pass
- ✅ Zero warnings (clippy + compiler)
- ✅ Documentation complete

---

**We're ~90% done with Phase 1. Just need working quantization to cross the finish line!** 🚀

**Recommendation**: Implement Q4_K support (1-2 hours) as the most robust path forward.
