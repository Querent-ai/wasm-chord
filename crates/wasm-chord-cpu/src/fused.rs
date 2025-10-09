//! Fused kernel operations for better performance
//!
//! These kernels combine multiple operations to reduce memory bandwidth
//! and improve cache locality.

use wasm_chord_core::error::Result;

/// Fused dequantization + matrix multiplication for Q4_K format
///
/// Combines dequantization and matmul into a single operation to:
/// - Reduce memory bandwidth (don't store intermediate dequantized values)
/// - Improve cache locality
/// - Reduce kernel launch overhead
pub fn fused_dequant_matmul_q4k(
    quantized: &[u8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    m: usize, // output rows
    n: usize, // output cols (quantized cols)
    k: usize, // input features
) -> Result<()> {
    // Q4_K block size is 256
    const BLOCK_SIZE: usize = 256;
    let num_blocks = k / BLOCK_SIZE;

    // For each output row
    for i in 0..m {
        // For each output column (corresponds to quantized matrix rows)
        for j in 0..n {
            let mut sum = 0.0f32;

            // Process each block
            for block_idx in 0..num_blocks {
                let block_offset = (j * num_blocks + block_idx) * (BLOCK_SIZE / 2 + 12);
                let scale_idx = j * num_blocks + block_idx;

                if scale_idx >= scales.len() {
                    continue;
                }

                let scale = scales[scale_idx];

                // Process values in the block
                for val_idx in 0..BLOCK_SIZE / 2 {
                    let byte_idx = block_offset + 12 + val_idx;
                    if byte_idx >= quantized.len() {
                        break;
                    }

                    let packed = quantized[byte_idx];
                    let v0 = (packed & 0x0F) as i8 - 8;
                    let v1 = ((packed >> 4) & 0x0F) as i8 - 8;

                    let input_idx0 = block_idx * BLOCK_SIZE + val_idx * 2;
                    let input_idx1 = input_idx0 + 1;

                    if input_idx0 < k {
                        sum += (v0 as f32) * scale * input[i * k + input_idx0];
                    }
                    if input_idx1 < k {
                        sum += (v1 as f32) * scale * input[i * k + input_idx1];
                    }
                }
            }

            output[i * n + j] = sum;
        }
    }

    Ok(())
}

/// Fused RMSNorm + Linear transformation
///
/// Combines normalization and matrix multiplication:
/// output = (input / rms(input)) * weight
pub fn fused_rmsnorm_linear(
    input: &[f32],
    weight: &[f32],
    norm_weight: &[f32],
    output: &mut [f32],
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    // Compute RMS
    let mut sum_sq = 0.0f32;
    for &val in input.iter().take(hidden_size) {
        sum_sq += val * val;
    }
    let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Fused normalize + matmul
    for i in 0..hidden_size {
        let mut sum = 0.0f32;
        for j in 0..hidden_size {
            let normalized = input[j] * inv_rms * norm_weight[j];
            sum += normalized * weight[i * hidden_size + j];
        }
        output[i] = sum;
    }

    Ok(())
}

/// Fused activation + projection
///
/// Combines SwiGLU activation with output projection:
/// output = (SiLU(gate) * up) @ down
pub fn fused_swiglu_proj(
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    output: &mut [f32],
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<()> {
    // Temporary for activated values
    let mut activated = vec![0.0f32; intermediate_size];

    // Fused SwiGLU activation
    for i in 0..intermediate_size {
        let x = gate[i];
        let silu = x / (1.0 + (-x).exp());
        activated[i] = silu * up[i];
    }

    // Project back to hidden size
    for i in 0..hidden_size {
        let mut sum = 0.0f32;
        for j in 0..intermediate_size {
            sum += activated[j] * down[i * intermediate_size + j];
        }
        output[i] = sum;
    }

    Ok(())
}

/// Fused attention scoring + softmax
///
/// Combines Q·K^T, scaling, masking, and softmax in one pass
pub fn fused_attention_score(
    query: &[f32],
    key: &[f32],
    _output: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal_mask: bool,
) -> Result<Vec<f32>> {
    let mut scores = vec![0.0f32; seq_len * seq_len];

    // Compute scores with scaling and masking
    for i in 0..seq_len {
        let mut max_score = f32::NEG_INFINITY;

        // Compute Q·K^T with scaling
        for j in 0..seq_len {
            if causal_mask && j > i {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
                continue;
            }

            let mut score = 0.0f32;
            for k in 0..head_dim {
                score += query[i * head_dim + k] * key[j * head_dim + k];
            }
            score *= scale;
            scores[i * seq_len + j] = score;

            if score > max_score {
                max_score = score;
            }
        }

        // Fused softmax (numerically stable)
        let mut sum_exp = 0.0f32;
        for j in 0..seq_len {
            if causal_mask && j > i {
                scores[i * seq_len + j] = 0.0;
                continue;
            }
            let exp_val = (scores[i * seq_len + j] - max_score).exp();
            scores[i * seq_len + j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        let inv_sum = 1.0 / sum_exp;
        for j in 0..seq_len {
            scores[i * seq_len + j] *= inv_sum;
        }
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_rmsnorm_linear() -> Result<()> {
        let hidden_size = 4;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let norm_weight = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![
            1.0, 0.0, 0.0, 0.0, // identity-like matrix
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut output = vec![0.0f32; hidden_size];

        fused_rmsnorm_linear(&input, &weight, &norm_weight, &mut output, hidden_size, 1e-6)?;

        // Output should be roughly normalized version of input
        assert!(output[0] > 0.0);
        assert!(output[1] > output[0]); // scaled values maintain relative order
        assert!(output[2] > output[1]);
        assert!(output[3] > output[2]);

        Ok(())
    }

    #[test]
    fn test_fused_swiglu_proj() -> Result<()> {
        let hidden_size = 4;
        let intermediate_size = 8;

        let gate = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
        let up = vec![1.0; intermediate_size];
        let down = vec![0.125f32; hidden_size * intermediate_size]; // average pooling

        let mut output = vec![0.0f32; hidden_size];

        fused_swiglu_proj(&gate, &up, &down, &mut output, hidden_size, intermediate_size)?;

        // All outputs should be non-zero (SwiGLU produces non-zero for these inputs)
        for &val in &output {
            assert!(val.abs() > 1e-6, "Output should be non-zero, got {}", val);
        }

        Ok(())
    }

    #[test]
    fn test_fused_attention_score_no_mask() -> Result<()> {
        let seq_len = 3;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let query = vec![
            1.0, 0.0, 0.0, 0.0, // q1
            0.0, 1.0, 0.0, 0.0, // q2
            0.0, 0.0, 1.0, 0.0, // q3
        ];
        let key = query.clone(); // same as query for simplicity

        let scores = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, false)?;

        // Check dimensions
        assert_eq!(scores.len(), seq_len * seq_len);

        // Each row should sum to ~1.0 (softmax property)
        for i in 0..seq_len {
            let row_sum: f32 = (0..seq_len).map(|j| scores[i * seq_len + j]).sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum: {}", i, row_sum);
        }

        Ok(())
    }

    #[test]
    fn test_fused_attention_score_with_causal_mask() -> Result<()> {
        let seq_len = 4;
        let head_dim = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let query = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let key = query.clone();

        let scores = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, true)?;

        // Check causal masking: upper triangle should be zero
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                assert_eq!(scores[i * seq_len + j], 0.0, "Causal mask failed at ({}, {})", i, j);
            }
        }

        // Check softmax on visible positions
        for i in 0..seq_len {
            let row_sum: f32 = (0..=i).map(|j| scores[i * seq_len + j]).sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum: {}", i, row_sum);
        }

        Ok(())
    }

    #[test]
    fn test_fused_dequant_matmul_q4k_basic() -> Result<()> {
        // Simple test with minimal data
        let m = 2; // output rows
        let n = 1; // output cols
        let k = 256; // must be multiple of block size

        // Create minimal quantized data (simplified)
        let quantized = vec![0u8; (k / 2 + 12) * n];
        let scales = vec![1.0f32; n];
        let input = vec![1.0f32; m * k];
        let mut output = vec![0.0f32; m * n];

        let result = fused_dequant_matmul_q4k(&quantized, &scales, &input, &mut output, m, n, k);
        assert!(result.is_ok());

        // Output should be computed (exact values depend on quantization)
        // Just verify no NaN/Inf
        for &val in &output {
            assert!(val.is_finite(), "Output should be finite, got {}", val);
        }

        Ok(())
    }

    #[test]
    fn test_fused_attention_score_properties() -> Result<()> {
        let seq_len = 5;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Random-ish query and key
        let query: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 % 3.0) - 1.0).collect();
        let key: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 % 5.0) - 2.0).collect();

        let scores = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, false)?;

        // All scores should be in [0, 1] (softmax output)
        for &score in &scores {
            assert!((0.0..=1.0).contains(&score), "Score out of range: {}", score);
        }

        // All scores should be finite
        for &score in &scores {
            assert!(score.is_finite(), "Score not finite: {}", score);
        }

        Ok(())
    }
}
