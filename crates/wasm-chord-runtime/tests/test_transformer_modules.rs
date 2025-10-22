//! Comprehensive tests for the refactored transformer modules

use wasm_chord_runtime::{
    attention::AttentionBackend, AttentionWeights, FFNWeights, FeedForward, GenerationConfig,
    KVCache, Model, MultiHeadAttention, TransformerConfig, TransformerLayer,
};

/// Test TransformerConfig creation and defaults
#[test]
fn test_config_default() {
    let config = TransformerConfig::default();
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_layers, 22);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_kv_heads, 4);
}

/// Test GenerationConfig creation and defaults
#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_p, 0.9);
    assert_eq!(config.top_k, 40);
    assert_eq!(config.repetition_penalty, 1.1);
}

/// Test KVCache creation and basic operations
#[test]
fn test_kv_cache_creation() {
    let max_seq_len = 128;
    let num_kv_heads = 4;
    let head_dim = 64;

    let cache = KVCache::new(max_seq_len, num_kv_heads, head_dim);

    assert_eq!(cache.current_seq_len, 0);
    assert_eq!(cache.max_seq_len, max_seq_len);
    assert_eq!(cache.num_kv_heads, num_kv_heads);
    assert_eq!(cache.head_dim, head_dim);
    assert_eq!(cache.keys.len(), max_seq_len * num_kv_heads * head_dim);
    assert_eq!(cache.values.len(), max_seq_len * num_kv_heads * head_dim);
}

/// Test KVCache clear operation
#[test]
fn test_kv_cache_clear() {
    let mut cache = KVCache::new(128, 4, 64);

    // Fill with some data
    let keys = vec![1.0; 4 * 64]; // 1 token
    let values = vec![2.0; 4 * 64];
    let _ = cache.append(&keys, &values).unwrap();

    assert_eq!(cache.current_seq_len, 1);

    // Clear cache
    cache.clear();

    assert_eq!(cache.current_seq_len, 0);
}

/// Test KVCache append operation
#[test]
fn test_kv_cache_append() {
    let mut cache = KVCache::new(128, 4, 64);

    // Append first token
    let keys1 = vec![1.0; 4 * 64]; // 1 token
    let values1 = vec![2.0; 4 * 64];
    let (cached_k, cached_v) = cache.append(&keys1, &values1).unwrap();

    assert_eq!(cache.current_seq_len, 1);
    assert_eq!(cached_k.len(), 4 * 64);
    assert_eq!(cached_v.len(), 4 * 64);

    // Append second token
    let keys2 = vec![3.0; 4 * 64];
    let values2 = vec![4.0; 4 * 64];
    let (cached_k, cached_v) = cache.append(&keys2, &values2).unwrap();

    assert_eq!(cache.current_seq_len, 2);
    assert_eq!(cached_k.len(), 2 * 4 * 64); // Now has 2 tokens
    assert_eq!(cached_v.len(), 2 * 4 * 64);
}

/// Test KVCache overflow handling
#[test]
fn test_kv_cache_overflow() {
    let mut cache = KVCache::new(2, 4, 64); // Small cache

    // Fill cache
    let keys = vec![1.0; 4 * 64];
    let values = vec![2.0; 4 * 64];
    cache.append(&keys, &values).unwrap();
    cache.append(&keys, &values).unwrap();

    // Try to overflow
    let result = cache.append(&keys, &values);
    assert!(result.is_err());
}

/// Test AttentionWeights creation
#[test]
fn test_attention_weights_creation() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let weights = AttentionWeights::new(&config);

    // Check dimensions
    assert_eq!(weights.wq.len(), config.hidden_size * config.hidden_size);
    assert_eq!(
        weights.wk.len(),
        config.hidden_size * config.num_kv_heads * (config.hidden_size / config.num_heads)
    );
    assert_eq!(
        weights.wv.len(),
        config.hidden_size * config.num_kv_heads * (config.hidden_size / config.num_heads)
    );
    assert_eq!(weights.wo.len(), config.hidden_size * config.hidden_size);
}

/// Test FFNWeights creation
#[test]
fn test_ffn_weights_creation() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let weights = FFNWeights::new(&config);

    // Check dimensions
    assert_eq!(weights.w_gate.len(), config.hidden_size * config.intermediate_size);
    assert_eq!(weights.w_up.len(), config.hidden_size * config.intermediate_size);
    assert_eq!(weights.w_down.len(), config.intermediate_size * config.hidden_size);
}

/// Test MultiHeadAttention creation
#[test]
fn test_multi_head_attention_creation() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let attn = MultiHeadAttention::new(config.clone());

    // Check head_dim is correctly computed
    assert_eq!(attn.config.hidden_size / attn.config.num_heads, 64);
}

/// Test FeedForward creation
#[test]
fn test_feed_forward_creation() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let _ffn = FeedForward::new(config);
    // Just test that it creates without panicking
}

/// Test TransformerLayer creation
#[test]
fn test_transformer_layer_creation() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let layer = TransformerLayer::new(&config);

    // Check that RMS norm weights are initialized
    assert_eq!(layer.ffn_norm.len(), 512);
}

/// Test Model creation
#[test]
fn test_model_creation() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let model = Model::new(config.clone());

    // Check basic structure
    assert_eq!(model.config.vocab_size, config.vocab_size);
    assert_eq!(model.config.hidden_size, config.hidden_size);
    assert_eq!(model.config.num_layers, config.num_layers);
}

/// Test Model RMS normalization
#[test]
fn test_model_rms_norm() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_size: 8,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let model = Model::new(config);

    // Test with simple values
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];

    let result = model.rms_norm(&input, &weight).unwrap();

    // Check output shape
    assert_eq!(result.len(), 4);

    // RMS norm should normalize the input
    let sum_sq: f32 = result.iter().map(|x| x * x).sum();
    let expected_sum_sq = input.len() as f32; // After normalization, sum of squares should be close to length

    // Allow some tolerance for floating point arithmetic
    assert!(
        (sum_sq - expected_sum_sq).abs() < 1.0,
        "RMS norm didn't properly normalize: sum_sq={}, expected={}",
        sum_sq,
        expected_sum_sq
    );
}

/// Test Model KV cache clearing
#[test]
fn test_model_clear_kv_cache() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let mut model = Model::new(config);

    // Clear cache (should not panic)
    model.clear_kv_cache();

    // Check that all caches are cleared
    for cache in &model.kv_caches {
        assert_eq!(cache.current_seq_len, 0);
    }
}

/// Test that config values are preserved through cloning
#[test]
fn test_config_clone() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let cloned = config.clone();

    assert_eq!(config.vocab_size, cloned.vocab_size);
    assert_eq!(config.hidden_size, cloned.hidden_size);
    assert_eq!(config.num_layers, cloned.num_layers);
    assert_eq!(config.num_heads, cloned.num_heads);
    assert_eq!(config.num_kv_heads, cloned.num_kv_heads);
    assert_eq!(config.intermediate_size, cloned.intermediate_size);
    assert_eq!(config.max_seq_len, cloned.max_seq_len);
    assert_eq!(config.rms_norm_eps, cloned.rms_norm_eps);
    assert_eq!(config.rope_theta, cloned.rope_theta);
}

/// Test that generation config values are preserved through cloning
#[test]
fn test_generation_config_clone() {
    let config = GenerationConfig {
        max_tokens: 50,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 50,
        repetition_penalty: 1.2,
    };

    let cloned = config.clone();

    assert_eq!(config.max_tokens, cloned.max_tokens);
    assert_eq!(config.temperature, cloned.temperature);
    assert_eq!(config.top_p, cloned.top_p);
    assert_eq!(config.top_k, cloned.top_k);
    assert_eq!(config.repetition_penalty, cloned.repetition_penalty);
}

/// Test KV cache with multiple tokens
#[test]
fn test_kv_cache_multiple_tokens() {
    let max_seq_len = 10;
    let num_kv_heads = 2;
    let head_dim = 4;
    let mut cache = KVCache::new(max_seq_len, num_kv_heads, head_dim);

    // Add multiple tokens one by one
    for i in 0..5 {
        let keys = vec![i as f32; num_kv_heads * head_dim];
        let values = vec![i as f32 + 10.0; num_kv_heads * head_dim];
        let (cached_k, cached_v) = cache.append(&keys, &values).unwrap();

        assert_eq!(cache.current_seq_len, i + 1);
        assert_eq!(cached_k.len(), (i + 1) * num_kv_heads * head_dim);
        assert_eq!(cached_v.len(), (i + 1) * num_kv_heads * head_dim);
    }

    // Get current data without appending
    let (k, v) = cache.current_data();
    assert_eq!(k.len(), 5 * num_kv_heads * head_dim);
    assert_eq!(v.len(), 5 * num_kv_heads * head_dim);
}

/// Test that model layers are properly initialized
#[test]
fn test_model_layers_initialization() {
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 512,
        num_layers: 3,
        num_heads: 8,
        num_kv_heads: 4,
        intermediate_size: 2048,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };

    let model = Model::new(config.clone());

    // Check that we have the correct number of layers
    assert_eq!(model.layers.len(), config.num_layers);

    // Check that we have the correct number of KV caches
    assert_eq!(model.kv_caches.len(), config.num_layers);
}
