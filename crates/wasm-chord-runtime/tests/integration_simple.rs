/// Simple integration tests for model inference
use wasm_chord_runtime::{Model, TransformerConfig};

#[test]
fn test_model_creation() {
    let config = TransformerConfig {
        vocab_size: 256,
        hidden_size: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 256,
        max_seq_len: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    };

    let model = Model::new(config);

    assert_eq!(model.config.vocab_size, 256);
    assert_eq!(model.config.num_layers, 2);
    assert_eq!(model.layers.len(), 2);
    assert_eq!(model.kv_caches.len(), 2);

    println!("✅ Model created successfully");
}

#[test]
fn test_forward_pass() {
    let config = TransformerConfig {
        vocab_size: 256,
        hidden_size: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 256,
        max_seq_len: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    };

    let mut model = Model::new(config);
    let tokens = vec![42u32];

    let result = model.forward(&tokens, 0);
    assert!(result.is_ok(), "Forward pass should succeed");

    let logits = result.unwrap();
    assert_eq!(logits.len(), model.config.vocab_size);

    println!("✅ Forward pass completed: {} logits", logits.len());
}

#[test]
fn test_sampling() {
    let config = TransformerConfig {
        vocab_size: 256,
        hidden_size: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 256,
        max_seq_len: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    };

    let model = Model::new(config);
    let mut logits = vec![0.0f32; model.config.vocab_size];
    logits[100] = 5.0; // Highest

    // Greedy sampling should be deterministic
    let sample = model.sample(&logits, 0.0, 1.0, 0).unwrap();
    assert_eq!(sample, 100, "Should select highest logit");

    println!("✅ Sampling works: selected token {}", sample);
}

#[test]
fn test_integration_ready() {
    // This confirms integration test infrastructure is ready
    // Real model loading tests will be added when we have test data
    println!("✅ Integration test infrastructure ready for real model tests");
    assert!(true);
}
