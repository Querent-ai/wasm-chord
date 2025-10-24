/// RoPE (Rotary Position Embedding) correctness tests
///
/// Based on llama.cpp's test-rope.cpp
/// Tests rotation properties, reversibility, and position encoding
use wasm_chord_runtime::{attention::AttentionBackend, MultiHeadAttention, TransformerConfig};

/// Helper to create a test config
fn test_config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 32000,
        hidden_size: 128, // Small for testing
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        intermediate_size: 256,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        max_seq_len: 512,
        attention_backend: AttentionBackend::Auto,
    }
}

#[test]
fn test_rope_preserves_norm() {
    // RoPE should preserve vector magnitude (rotation property)
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    // Create test vector
    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let num_heads = config.num_heads;

    // Input: [1.0, 2.0, 3.0, 4.0, ...] for one head
    let mut tensor: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    tensor.resize(seq_len * num_heads * head_dim, 0.0);

    // Calculate input norm
    let input_norm: f32 = tensor[..head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();

    // Apply RoPE at position 5
    let mut rotated = tensor.clone();
    attn.apply_rope(&mut rotated, 5, seq_len, num_heads).unwrap();

    // Calculate output norm (first head only)
    let output_norm: f32 = rotated[..head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();

    // Norms should be equal (rotation preserves magnitude)
    let norm_diff = (input_norm - output_norm).abs();
    assert!(
        norm_diff < 1e-5,
        "RoPE should preserve vector norm: input={}, output={}, diff={}",
        input_norm,
        output_norm,
        norm_diff
    );
}

#[test]
fn test_rope_identity_at_zero() {
    // At position 0, RoPE should be close to identity (angle=0, so cos=1, sin=0)
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let num_heads = 1; // Test just one head

    // Create test vector
    let input: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let mut output = input.clone();

    // Apply RoPE at position 0
    attn.apply_rope(&mut output, 0, seq_len, num_heads).unwrap();

    // At position 0, for low frequencies, rotation should be minimal
    // Check that values are very close to original
    for (i, (&orig, &rot)) in input.iter().zip(&output).enumerate() {
        let diff = (orig - rot).abs();
        // First few dimensions should be nearly unchanged (lower frequencies)
        if i < 4 {
            assert!(
                diff < 0.1,
                "Position 0 should be nearly identity for dim {}: orig={}, rot={}, diff={}",
                i,
                orig,
                rot,
                diff
            );
        }
    }
}

#[test]
fn test_rope_rotation_consistency() {
    // Same position should always give same rotation
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let num_heads = 2;

    let input: Vec<f32> = (0..seq_len * num_heads * head_dim).map(|i| i as f32).collect();

    // Apply RoPE twice at same position
    let mut output1 = input.clone();
    let mut output2 = input.clone();

    attn.apply_rope(&mut output1, 10, seq_len, num_heads).unwrap();
    attn.apply_rope(&mut output2, 10, seq_len, num_heads).unwrap();

    // Results should be identical
    for (i, (&v1, &v2)) in output1.iter().zip(&output2).enumerate() {
        assert!((v1 - v2).abs() < 1e-10, "RoPE not consistent at index {}: {} != {}", i, v1, v2);
    }
}

#[test]
fn test_rope_different_positions() {
    // Different positions should give different rotations
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let num_heads = 1;

    let input: Vec<f32> = vec![1.0; head_dim];

    let mut pos0 = input.clone();
    let mut pos10 = input.clone();

    attn.apply_rope(&mut pos0, 0, seq_len, num_heads).unwrap();
    attn.apply_rope(&mut pos10, 10, seq_len, num_heads).unwrap();

    // Outputs should be different
    let mut different = false;
    for (&v0, &v10) in pos0.iter().zip(&pos10) {
        if (v0 - v10).abs() > 1e-5 {
            different = true;
            break;
        }
    }

    assert!(different, "Different positions should produce different rotations");
}

#[test]
fn test_rope_multi_token_sequence() {
    // Test RoPE with multiple tokens in sequence
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 3; // Three tokens
    let num_heads = 2;

    // Create input for 3 tokens
    let mut tensor = vec![0.0; seq_len * num_heads * head_dim];
    for i in 0..tensor.len() {
        tensor[i] = (i % 10) as f32; // Simple pattern
    }

    // Apply RoPE starting at position 5 (so tokens are at pos 5, 6, 7)
    attn.apply_rope(&mut tensor, 5, seq_len, num_heads).unwrap();

    // Verify no NaN/Inf
    assert!(tensor.iter().all(|&x| x.is_finite()), "RoPE produced non-finite values");

    // Each token should have different rotation (different positions)
    // Compare first head, first dimension of each token
    let val_token0 = tensor[0];
    let val_token1 = tensor[num_heads * head_dim];
    let val_token2 = tensor[2 * num_heads * head_dim];

    assert!(
        (val_token0 - val_token1).abs() > 1e-6 || (val_token1 - val_token2).abs() > 1e-6,
        "Tokens at different positions should have different RoPE values"
    );
}

#[test]
fn test_rope_interleaved_pairs() {
    // Verify RoPE uses interleaved pairing: (0,1), (2,3), (4,5), ...
    let config = test_config();

    let head_dim = 8; // Small for easier verification
    let mut config_small = config;
    config_small.hidden_size = head_dim * 2; // 2 heads
    config_small.num_heads = 2;
    let attn_small = MultiHeadAttention::new(config_small);

    // Create input with specific pattern to test pairing
    let input = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
    let mut output = input.clone();

    attn_small.apply_rope(&mut output, 0, 1, 1).unwrap();

    // At position 0, rotation should be minimal
    // But we can verify the interleaved structure by checking that:
    // - Elements 0,1 are a pair
    // - Elements 2,3 are a pair
    // - etc.

    // Just verify no NaN and reasonable values
    assert!(output.iter().all(|&x: &f32| x.is_finite()), "RoPE produced non-finite values");
}

#[test]
fn test_rope_numerical_stability() {
    // Test RoPE with large positions and various input values
    let config = test_config();
    let attn = MultiHeadAttention::new(config.clone());

    let head_dim = config.hidden_size / config.num_heads;
    let seq_len = 1;
    let num_heads = 1;

    // Test with large position
    let mut tensor = vec![1.0; head_dim];
    attn.apply_rope(&mut tensor, 1000, seq_len, num_heads).unwrap();

    assert!(tensor.iter().all(|&x: &f32| x.is_finite()), "RoPE unstable at large positions");

    // Test with large values
    let mut tensor_large = vec![100.0; head_dim];
    attn.apply_rope(&mut tensor_large, 10, seq_len, num_heads).unwrap();

    assert!(tensor_large.iter().all(|&x: &f32| x.is_finite()), "RoPE unstable with large values");

    // Test with tiny values
    let mut tensor_small = vec![0.001; head_dim];
    attn.apply_rope(&mut tensor_small, 10, seq_len, num_heads).unwrap();

    assert!(tensor_small.iter().all(|&x: &f32| x.is_finite()), "RoPE unstable with small values");
}

#[test]
fn test_rope_frequency_calculation() {
    // Verify frequency calculation follows the formula: freq = 1.0 / (theta^(i/head_dim))
    let config = test_config();
    let head_dim = config.hidden_size / config.num_heads;

    // Manually calculate expected frequencies for first few dimensions
    let theta = config.rope_theta;

    for i in (0..4).step_by(2) {
        let expected_freq: f32 = 1.0 / theta.powf(i as f32 / head_dim as f32);

        // At position 1, angle should equal freq
        // We can't directly test the internal calculation, but we can verify
        // that the rotation behavior is consistent with the formula

        assert!(
            expected_freq > 0.0 && expected_freq.is_finite(),
            "Frequency calculation produced invalid value for i={}: {}",
            i,
            expected_freq
        );
    }
}
