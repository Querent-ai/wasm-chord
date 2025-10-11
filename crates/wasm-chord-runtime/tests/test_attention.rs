/// Attention mechanism correctness tests
///
/// Tests scaled dot-product attention, softmax properties, and causal masking
use wasm_chord_runtime::{MultiHeadAttention, TransformerConfig};

fn test_config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 32000,
        hidden_size: 64, // Small for testing
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        intermediate_size: 128,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        max_seq_len: 512,
    }
}

#[test]
fn test_attention_softmax_sum() {
    // Attention weights should sum to 1 (softmax property)
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let kv_seq_len = 3; // Attend to 3 tokens

    // Create Q: one query vector
    let q = vec![1.0; seq_len * config.num_heads * head_dim];

    // Create K, V: three key/value vectors
    let k = vec![0.5; kv_seq_len * config.num_kv_heads * head_dim];
    let v = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];

    // Compute attention
    let output = attn.compute_attention(&q, &k, &v, seq_len, 0).unwrap();

    // Output should be valid (no NaN/Inf)
    assert!(output.iter().all(|&x| x.is_finite()), "Attention produced non-finite values");

    // With uniform V, output should be close to V values (weighted average)
    // Since all V are 1.0, output should be close to 1.0
    let avg_output: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        (avg_output - 1.0).abs() < 0.5,
        "Attention output should be close to average of V: got {}",
        avg_output
    );
}

#[test]
fn test_attention_with_uniform_values() {
    // When Q=K for all tokens, attention should be uniform (accounting for causal mask)
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let kv_seq_len = 2;

    // Q and K are identical
    let q = vec![1.0; seq_len * config.num_heads * head_dim];
    let k = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];

    // V has different values for each token
    let mut v = vec![0.0; kv_seq_len * config.num_kv_heads * head_dim];
    for i in 0..kv_seq_len {
        let offset = i * config.num_kv_heads * head_dim;
        v[offset..offset + head_dim].fill((i + 1) as f32);
    }

    // Query at position 1 so it can attend to both tokens (0 and 1)
    let output = attn.compute_attention(&q, &k, &v, seq_len, 1).unwrap();

    // Since Q·K is same for all tokens, attention weights should be uniform
    // Output should be approximately the average of V values: (1 + 2) / 2 = 1.5
    let first_head_output: f32 = output[..head_dim].iter().sum::<f32>() / head_dim as f32;

    assert!(
        (first_head_output - 1.5).abs() < 0.1,
        "Uniform Q·K should give uniform attention (average V): got {}",
        first_head_output
    );
}

#[test]
fn test_attention_causal_mask() {
    // Future tokens should be masked (causal attention)
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let kv_seq_len = 5; // 5 cached tokens

    let q = vec![1.0; seq_len * config.num_heads * head_dim];
    let k = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];

    // Create V with distinct values for each position
    let mut v = vec![0.0; kv_seq_len * config.num_kv_heads * head_dim];
    for i in 0..kv_seq_len {
        v[i * config.num_kv_heads * head_dim..][..head_dim].fill((i + 1) as f32 * 10.0);
    }

    // Query at position 2 (should only attend to positions 0, 1, 2)
    let output = attn.compute_attention(&q, &k, &v, seq_len, 2).unwrap();

    // Output should be average of first 3 positions: (10 + 20 + 30) / 3 = 20
    let first_head_output: f32 = output[..head_dim].iter().sum::<f32>() / head_dim as f32;

    assert!(
        (first_head_output - 20.0).abs() < 5.0,
        "Causal masking should only attend to past: expected ~20, got {}",
        first_head_output
    );
}

#[test]
fn test_attention_scaling() {
    // Attention scores should be scaled by 1/sqrt(head_dim)
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let kv_seq_len = 2;

    // Create Q and K with large dot product
    let q = vec![10.0; seq_len * config.num_heads * head_dim];
    let k = vec![10.0; kv_seq_len * config.num_kv_heads * head_dim];
    let v = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];

    let output = attn.compute_attention(&q, &k, &v, seq_len, 0).unwrap();

    // With scaling, output should still be reasonable (not explode)
    assert!(
        output.iter().all(|&x| x.is_finite() && x.abs() < 100.0),
        "Attention scaling should prevent explosion"
    );
}

#[test]
fn test_attention_numerical_stability() {
    // Test attention with extreme values
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let kv_seq_len = 3;

    // Test with very large values
    let q_large = vec![1000.0; seq_len * config.num_heads * head_dim];
    let k_large = vec![1000.0; kv_seq_len * config.num_kv_heads * head_dim];
    let v_large = vec![100.0; kv_seq_len * config.num_kv_heads * head_dim];

    let output = attn.compute_attention(&q_large, &k_large, &v_large, seq_len, 0).unwrap();

    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Attention should be numerically stable with large values"
    );

    // Test with very small values
    let q_small = vec![0.001; seq_len * config.num_heads * head_dim];
    let k_small = vec![0.001; kv_seq_len * config.num_kv_heads * head_dim];
    let v_small = vec![0.001; kv_seq_len * config.num_kv_heads * head_dim];

    let output = attn.compute_attention(&q_small, &k_small, &v_small, seq_len, 0).unwrap();

    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Attention should be numerically stable with small values"
    );
}

#[test]
fn test_attention_output_shape() {
    // Output should have same shape as Q
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 2;
    let kv_seq_len = 5;

    let q = vec![1.0; seq_len * config.num_heads * head_dim];
    let k = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];
    let v = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];

    let output = attn.compute_attention(&q, &k, &v, seq_len, 0).unwrap();

    assert_eq!(output.len(), q.len(), "Attention output should have same shape as query");
}

#[test]
fn test_attention_deterministic() {
    // Same input should give same output
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 2;
    let kv_seq_len = 3;

    let q = (0..seq_len * config.num_heads * head_dim).map(|i| (i % 10) as f32).collect::<Vec<_>>();
    let k = (0..kv_seq_len * config.num_kv_heads * head_dim)
        .map(|i| ((i * 3) % 10) as f32)
        .collect::<Vec<_>>();
    let v = (0..kv_seq_len * config.num_kv_heads * head_dim)
        .map(|i| ((i * 7) % 10) as f32)
        .collect::<Vec<_>>();

    let output1 = attn.compute_attention(&q, &k, &v, seq_len, 1).unwrap();
    let output2 = attn.compute_attention(&q, &k, &v, seq_len, 1).unwrap();

    for (i, (&v1, &v2)) in output1.iter().zip(&output2).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "Attention not deterministic at index {}: {} != {}",
            i,
            v1,
            v2
        );
    }
}

#[test]
fn test_attention_gqa_repeat() {
    // Test Grouped Query Attention (GQA) with num_heads > num_kv_heads
    let mut config = test_config();
    config.num_heads = 8;
    config.num_kv_heads = 2; // GQA: 8 query heads, 2 KV heads

    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let kv_seq_len = 2;

    let q = vec![1.0; seq_len * config.num_heads * head_dim];
    let k = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];
    let v = vec![1.0; kv_seq_len * config.num_kv_heads * head_dim];

    let output = attn.compute_attention(&q, &k, &v, seq_len, 0).unwrap();

    // Should work without errors
    assert_eq!(output.len(), seq_len * config.num_heads * head_dim, "GQA output shape incorrect");

    assert!(output.iter().all(|&x| x.is_finite()), "GQA produced non-finite values");
}

#[test]
fn test_attention_single_token() {
    // Test attention with single token (common during generation)
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let kv_seq_len = 1;

    let q = vec![2.0; seq_len * config.num_heads * head_dim];
    let k = vec![2.0; kv_seq_len * config.num_kv_heads * head_dim];
    let v = vec![5.0; kv_seq_len * config.num_kv_heads * head_dim];

    let output = attn.compute_attention(&q, &k, &v, seq_len, 0).unwrap();

    // With single token, attention weight is 1.0, so output = V
    let avg_output: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        (avg_output - 5.0).abs() < 0.1,
        "Single token attention should output V: expected 5.0, got {}",
        avg_output
    );
}
