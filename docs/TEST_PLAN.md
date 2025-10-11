# Comprehensive Test Plan for wasm-chord

Based on llama.cpp test structure, here's a plan to add robust unit tests.

## Current Test Coverage

### ✅ Existing Tests (Good)
- Basic quantization (Q4_0, Q8_0, Q4_K, Q6_K, Q5_K, Q8_K) - check for NaN/Inf
- Tokenizer encode/decode
- Tensor operations (matmul, RMS norm, attention)
- GGUF parsing
- KV cache management

### ❌ Missing Tests (Need to Add)
- Quantization accuracy (RMSE against reference)
- RoPE correctness with known values
- Attention mechanism with known outputs
- Full forward pass with deterministic inputs
- Model loading completeness
- Numerical stability tests

---

## Phase 1: Quantization Accuracy Tests (Like llama.cpp)

### Test Structure
```rust
// crates/wasm-chord-core/src/quant.rs tests

/// Calculate RMSE between two float arrays
fn array_rmse(a1: &[f32], a2: &[f32]) -> f32 {
    assert_eq!(a1.len(), a2.len());
    let sum: f32 = a1.iter()
        .zip(a2)
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    (sum / a1.len() as f32).sqrt()
}

/// Generate synthetic test data (like llama.cpp)
fn generate_test_data(offset: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.1 + 2.0 * ((i as f32) + offset).cos())
        .collect()
}

#[test]
fn test_q4_k_quantization_accuracy() {
    // Generate test data
    let test_data = generate_test_data(0.0, 256);

    // Quantize and dequantize
    let mut block = BlockQ4_K::default();
    quantize_q4_k(&test_data, &mut block); // Need to implement

    let mut output = vec![0.0f32; 256];
    dequantize_q4_k(&block, &mut output).unwrap();

    // Check accuracy
    let rmse = array_rmse(&test_data, &output);

    // llama.cpp uses MAX_QUANTIZATION_TOTAL_ERROR = 0.002
    assert!(rmse < 0.01, "Q4_K RMSE too high: {}", rmse);
}
```

### Tests to Add
1. `test_q4_0_accuracy()` - RMSE < 0.002
2. `test_q4_k_accuracy()` - RMSE < 0.002
3. `test_q5_k_accuracy()` - RMSE < 0.002
4. `test_q6_k_accuracy()` - RMSE < 0.002
5. `test_q8_0_accuracy()` - RMSE < 0.001
6. `test_q8_k_accuracy()` - RMSE < 0.001

**Acceptance**: All RMSE values should be similar to llama.cpp's thresholds.

---

## Phase 2: RoPE (Rotary Position Embedding) Tests

### Test Structure
```rust
// crates/wasm-chord-runtime/src/transformer/tests/test_rope.rs

#[test]
fn test_rope_basic() {
    // Known input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let head_dim = 4;
    let position = 0;

    // Apply RoPE
    let output = apply_rope(&input, head_dim, position);

    // For position 0, RoPE should be identity (cos=1, sin=0)
    for (i, &val) in output.iter().enumerate() {
        assert!((val - input[i]).abs() < 1e-6);
    }
}

#[test]
fn test_rope_rotation() {
    let head_dim = 4;
    let input = vec![1.0, 0.0, 0.0, 1.0];

    // Apply at position 1
    let output = apply_rope(&input, head_dim, 1);

    // Verify rotation properties
    // ||output|| should equal ||input||
    let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
    let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!((input_norm - output_norm).abs() < 1e-5);
}

#[test]
fn test_rope_reversible() {
    // RoPE with -θ should reverse RoPE with θ
    let input = vec![1.0, 2.0, 3.0, 4.0];

    let forward = apply_rope(&input, 4, 5);
    let backward = apply_rope_inverse(&forward, 4, 5);

    for (i, &val) in backward.iter().enumerate() {
        assert!((val - input[i]).abs() < 1e-5);
    }
}
```

### Tests to Add
1. `test_rope_identity_at_zero()` - Position 0 is identity
2. `test_rope_preserves_norm()` - ||output|| = ||input||
3. `test_rope_rotation_angle()` - Correct rotation angles
4. `test_rope_reversible()` - Forward+inverse = identity

---

## Phase 3: Attention Mechanism Tests

### Test Structure
```rust
// crates/wasm-chord-runtime/src/transformer/tests/test_attention.rs

#[test]
fn test_attention_identity() {
    // Q = K, all equal -> uniform attention
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let k = vec![
        1.0, 0.0, 0.0, 0.0,  // Same as Q
        1.0, 0.0, 0.0, 0.0,
    ];
    let v = vec![
        5.0, 0.0, 0.0, 0.0,
        3.0, 0.0, 0.0, 0.0,
    ];

    let output = compute_attention(&q, &k, &v, 4, 2);

    // Should be average of values: (5 + 3) / 2 = 4
    assert!((output[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_attention_softmax_sum() {
    // Attention weights should sum to 1
    let q = random_vector(64);
    let k = random_matrix(64, 10);

    let weights = compute_attention_weights(&q, &k);
    let sum: f32 = weights.iter().sum();

    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_attention_causal_mask() {
    // With causal mask, future tokens should have zero weight
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let k = vec![
        1.0, 0.0, 0.0, 0.0,  // Position 0
        0.0, 1.0, 0.0, 0.0,  // Position 1 (should be masked)
    ];

    let weights = compute_attention_weights_with_mask(&q, &k, true);

    // Weight for position 1 should be 0
    assert!(weights[1].abs() < 1e-6);
}
```

### Tests to Add
1. `test_attention_softmax_properties()` - Weights sum to 1
2. `test_attention_with_uniform_values()` - Known output
3. `test_attention_causal_mask()` - Future masked correctly
4. `test_attention_numerical_stability()` - No overflow with large values

---

## Phase 4: Model Component Tests

### Test Structure
```rust
// crates/wasm-chord-runtime/tests/test_model_components.rs

#[test]
fn test_rmsnorm_properties() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weights = vec![1.0; 4];

    let output = rms_norm(&input, &weights, 1e-6);

    // Check RMS is approximately 1
    let rms: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt() / (output.len() as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-5);
}

#[test]
fn test_silu_activation() {
    // SiLU(x) = x * sigmoid(x)
    let x = 2.0f32;
    let silu_x = silu(x);

    let sigmoid_x = 1.0 / (1.0 + (-x).exp());
    let expected = x * sigmoid_x;

    assert!((silu_x - expected).abs() < 1e-6);
}

#[test]
fn test_layer_forward_deterministic() {
    // Same input should always give same output
    let input = vec![0.5; 2048];

    let output1 = layer_forward(&input, 0);
    let output2 = layer_forward(&input, 0);

    for (o1, o2) in output1.iter().zip(&output2) {
        assert!((o1 - o2).abs() < 1e-6);
    }
}
```

### Tests to Add
1. `test_rmsnorm_unit_rms()` - RMS is approximately 1
2. `test_silu_known_values()` - SiLU(x) = x * sigmoid(x)
3. `test_ffn_glu_gate()` - GLU gating works correctly
4. `test_layer_residual()` - Residual connections add correctly

---

## Phase 5: Integration Tests with Real Models

### Test Structure
```rust
// crates/wasm-chord-runtime/tests/test_known_outputs.rs

#[test]
fn test_tinyllama_first_token() {
    // Load TinyLlama model
    let model = load_model("models/tinyllama-1.1b.Q4_K_M.gguf");
    let tokenizer = load_tokenizer(&model);

    // Known prompt and expected token
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false).unwrap();

    // Run forward pass
    let logits = model.forward(&tokens).unwrap();

    // Get top token
    let top_token = argmax(&logits);

    // Compare against llama.cpp output (run once to establish ground truth)
    // This is a regression test - if we change the model, we should get same outputs
    let expected_token = 29892; // Comma token (from llama.cpp)

    assert_eq!(top_token, expected_token,
        "First token mismatch! This indicates a change in model behavior.");
}

#[test]
fn test_model_generation_deterministic() {
    // With temperature=0 (greedy), output should be deterministic
    let model = load_model("models/tinyllama-1.1b.Q4_K_M.gguf");

    let output1 = model.generate("Hello", temperature=0.0, max_tokens=5);
    let output2 = model.generate("Hello", temperature=0.0, max_tokens=5);

    assert_eq!(output1, output2, "Deterministic generation failed");
}
```

### Tests to Add
1. `test_known_logits()` - Match llama.cpp logits for specific prompt
2. `test_deterministic_generation()` - Same input = same output at temp=0
3. `test_model_loads_completely()` - All weights loaded, no NaN
4. `test_kv_cache_consistency()` - KV cache gives same results as without

---

## Phase 6: Performance / Regression Tests

### Test Structure
```rust
#[test]
#[ignore] // Run with --ignored flag
fn benchmark_quantization_speed() {
    let data = generate_test_data(0.0, 1_000_000);

    let start = std::time::Instant::now();
    for block in data.chunks(256) {
        let mut output = vec![0.0f32; 256];
        dequantize_q4_k(/* ... */);
    }
    let duration = start.elapsed();

    // Should process at least 100M elements/sec
    let elements_per_sec = 1_000_000.0 / duration.as_secs_f64();
    assert!(elements_per_sec > 100_000_000.0);
}
```

---

## Implementation Order

1. **Phase 1**: Quantization RMSE tests ✅ (Most important - verifies correctness)
2. **Phase 2**: RoPE tests (Critical for position encoding)
3. **Phase 3**: Attention tests (Core model component)
4. **Phase 4**: Model component tests (Layer, FFN, etc.)
5. **Phase 5**: Integration tests with known outputs (Regression protection)
6. **Phase 6**: Performance benchmarks (Optional, run with `--ignored`)

---

## File Structure

```
crates/wasm-chord-core/
└── src/
    └── quant.rs  (add RMSE tests inline)

crates/wasm-chord-runtime/
└── tests/
    ├── test_quantization_accuracy.rs  (Phase 1)
    ├── test_rope.rs                   (Phase 2)
    ├── test_attention.rs              (Phase 3)
    ├── test_model_components.rs       (Phase 4)
    ├── test_known_outputs.rs          (Phase 5)
    └── benchmarks.rs                  (Phase 6)
```

---

## Success Criteria

- ✅ All quantization RMSE values match llama.cpp thresholds
- ✅ RoPE tests verify rotation properties
- ✅ Attention tests pass with known inputs
- ✅ Model generates same outputs as llama.cpp for test prompts
- ✅ All tests pass on CI
- ✅ Test coverage > 80% for core modules

---

## Commands to Run

```bash
# Run all tests
cargo test --workspace

# Run specific test file
cargo test --test test_quantization_accuracy

# Run with verbose output
cargo test -- --nocapture

# Run ignored/slow tests
cargo test -- --ignored

# Run tests and show coverage
cargo tarpaulin --workspace
```
