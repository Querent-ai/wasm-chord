//! Multi-head attention implementation

use wasm_chord_core::error::Result;
use wasm_chord_cpu::{matmul_f32, matmul_transposed};

#[cfg(feature = "gpu")]
use wasm_chord_gpu::GpuBackend;

use super::{KVCache, TransformerConfig};

/// Multi-head attention layer
#[allow(dead_code)]
pub struct MultiHeadAttention {
    pub config: TransformerConfig,
    head_dim: usize,
}

#[allow(dead_code)]
impl MultiHeadAttention {
    pub fn new(config: TransformerConfig) -> Self {
        let head_dim = config.hidden_size / config.num_heads;
        Self { config, head_dim }
    }

    /// Helper: matrix multiplication with GPU/CPU fallback
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        transposed_b: bool,
        #[cfg(feature = "gpu")] gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu {
            if !transposed_b {
                if let Ok(result) = gpu.matmul(a, b, m as u32, k as u32, n as u32) {
                    return Ok(result);
                }
            }
        }

        // CPU fallback
        let mut result = vec![0.0; m * n];
        if transposed_b {
            matmul_transposed(a, b, &mut result, m, k, n)?;
        } else {
            matmul_f32(a, b, &mut result, m, k, n)?;
        }
        Ok(result)
    }

    /// Apply attention with KV caching
    ///
    /// # Arguments
    /// * `hidden_states` - Input [batch, seq_len, hidden_size]
    /// * `weights` - Model weights (wq, wk, wv, wo)
    /// * `kv_cache` - KV cache for this layer
    /// * `position` - Current position in sequence
    /// * `gpu` - Optional GPU backend
    ///
    /// # Returns
    /// Output [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &[f32],
        weights: &AttentionWeights,
        kv_cache: &mut KVCache,
        position: usize,
        #[cfg(feature = "gpu")] gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.config.hidden_size;
        let hidden_size = self.config.hidden_size;

        // Project to Q, K, V
        // Q projection: [seq_len, hidden_size] x [hidden_size, hidden_size]
        let q = self.matmul(
            hidden_states,
            &weights.wq,
            seq_len,
            hidden_size,
            hidden_size,
            true, // FIXED: GGUF stores weights in transposed format
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        // Debug Q weights and projection
        if std::env::var("DEBUG_WEIGHTS").is_ok() {
            eprintln!("  WQ weights (first 10): {:?}", &weights.wq[..10.min(weights.wq.len())]);
            eprintln!(
                "  WQ weights sum: {:.6}, mean: {:.6}",
                weights.wq.iter().sum::<f32>(),
                weights.wq.iter().sum::<f32>() / weights.wq.len() as f32
            );
            eprintln!("  Q projection (first 10): {:?}", &q[..10.min(q.len())]);
            eprintln!(
                "  Q projection sum: {:.6}, mean: {:.6}",
                q.iter().sum::<f32>(),
                q.iter().sum::<f32>() / q.len() as f32
            );
        }

        // K projection: [seq_len, hidden_size] x [hidden_size, num_kv_heads * head_dim]
        let k = self.matmul(
            hidden_states,
            &weights.wk,
            seq_len,
            hidden_size,
            self.config.num_kv_heads * self.head_dim,
            true, // FIXED: GGUF stores weights in transposed format
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        // V projection
        let v = self.matmul(
            hidden_states,
            &weights.wv,
            seq_len,
            hidden_size,
            self.config.num_kv_heads * self.head_dim,
            true, // FIXED: GGUF stores weights in transposed format
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        // DEBUG: Dump Q, K, V for layer 0
        if std::env::var("DUMP_LAYER0").is_ok() {
            eprintln!("LAYER0 Q preview: {:?}", &q[..q.len().min(16)]);
            eprintln!("LAYER0 K preview: {:?}", &k[..k.len().min(16)]);
            eprintln!("LAYER0 V preview: {:?}", &v[..v.len().min(16)]);
        }

        // Apply RoPE (Rotary Position Embedding)
        // Apply different positions for each token in the sequence
        let mut q = q;
        let mut k = k;
        self.apply_rope(&mut q, position, seq_len, self.config.num_heads)?;
        self.apply_rope(&mut k, position, seq_len, self.config.num_kv_heads)?;

        // MINIMAL KV CACHE: Use a very simple approach that's guaranteed to work
        let output = if std::env::var("DISABLE_KV").is_ok() {
            // Bypass cache: attend only to current K/V (useful for isolating KV issues)
            if std::env::var("DEBUG_KV").is_ok() {
                eprintln!(
                    "  DISABLE_KV active: skipping cache append; using in-batch K/V (len={})",
                    k.len()
                );
            }
            self.compute_attention(&q, &k, &v, seq_len, position)?
        } else {
            // Use KV cache properly - this is the key fix!
            if std::env::var("DEBUG_KV").is_ok() {
                eprintln!("  Using KV cache - appending new K/V to cache");
            }

            // Append new K/V to cache and get all cached K/V for attention
            let (cached_k, cached_v) = kv_cache.append(&k, &v)?;

            if std::env::var("DEBUG_KV").is_ok() {
                eprintln!(
                    "  After append: kv_cache.current_seq_len={}, cached_k.len()={}",
                    kv_cache.current_seq_len,
                    cached_k.len()
                );
            }

            // Use cached K/V for attention computation
            self.compute_attention(&q, &cached_k, &cached_v, seq_len, position)?
        };

        // DEBUG: Dump attention output for layer 0
        if std::env::var("DUMP_LAYER0").is_ok() {
            eprintln!("LAYER0 attn_output preview: {:?}", &output[..output.len().min(16)]);
        }

        // Output projection
        let result = self.matmul(
            &output,
            &weights.wo,
            seq_len,
            hidden_size,
            hidden_size,
            true, // FIXED: GGUF stores weights in transposed format
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        Ok(result)
    }

    /// Apply RoPE (Rotary Position Embedding) correctly for multiple tokens
    ///
    /// # Arguments
    /// * `tensor` - Q or K tensor with shape [seq_len, num_heads, head_dim]
    /// * `start_pos` - Starting position in the sequence
    /// * `seq_len` - Number of tokens in this batch
    /// * `num_heads` - Number of heads (num_heads for Q, num_kv_heads for K/V)
    fn apply_rope(
        &self,
        tensor: &mut [f32],
        start_pos: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Result<()> {
        let head_dim = self.head_dim;
        let debug_rope = std::env::var("DEBUG_ROPE").is_ok();

        // For each token in the sequence
        for seq_idx in 0..seq_len {
            let token_pos = start_pos + seq_idx;

            if debug_rope && seq_idx == 0 {
                eprintln!(
                    "  RoPE: token_pos={}, head_dim={}, theta={}",
                    token_pos, head_dim, self.config.rope_theta
                );
            }

            // For each head
            for head in 0..num_heads {
                let base_idx = (seq_idx * num_heads + head) * head_dim;

                // Apply RoPE using INTERLEAVED pairing: (0,1), (2,3), (4,5), ...
                // This matches llama2.c implementation
                for i in (0..head_dim).step_by(2) {
                    let idx0 = base_idx + i;
                    let idx1 = base_idx + i + 1;

                    // RoPE frequency calculation
                    // Since we step_by(2), i goes 0,2,4,6,... so we use i (not 2*i)
                    // freq = 1.0 / (theta ^ (i / head_dim))
                    let freq = 1.0 / self.config.rope_theta.powf(i as f32 / head_dim as f32);
                    let angle = token_pos as f32 * freq;

                    let cos = angle.cos();
                    let sin = angle.sin();

                    // Rotate the pair
                    let x0 = tensor[idx0];
                    let x1 = tensor[idx1];

                    let new_x0 = x0 * cos - x1 * sin;
                    let new_x1 = x0 * sin + x1 * cos;

                    tensor[idx0] = new_x0;
                    tensor[idx1] = new_x1;

                    if debug_rope && head == 0 && i < 4 {
                        eprintln!(
                            "    head={}, i={}, freq={:.6}, angle={:.6}, cos={:.6}, sin={:.6}",
                            head, i, freq, angle, cos, sin
                        );
                        eprintln!("      ({:.6}, {:.6}) -> ({:.6}, {:.6})", x0, x1, new_x0, new_x1);
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized dot product with manual loop unrolling
    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;

        // Process 4 elements at a time (loop unrolling)
        let chunks = len / 4;

        for i in 0..chunks {
            let idx = i * 4;
            sum += a[idx] * b[idx];
            sum += a[idx + 1] * b[idx + 1];
            sum += a[idx + 2] * b[idx + 2];
            sum += a[idx + 3] * b[idx + 3];
        }

        // Handle remaining elements
        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }

        sum
    }

    /// Repeat K/V tensors for Grouped Query Attention (GQA)
    ///
    /// Expands K/V from num_kv_heads to num_heads by repeating each KV head n_rep times.
    /// Input shape: [seq_len, num_kv_heads, head_dim]
    /// Output shape: [seq_len, num_heads, head_dim]
    fn repeat_kv(&self, kv: &[f32], seq_len: usize, n_rep: usize) -> Vec<f32> {
        if n_rep == 1 {
            return kv.to_vec();
        }

        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.head_dim;
        let mut output = vec![0.0; seq_len * num_kv_heads * n_rep * head_dim];

        for seq_idx in 0..seq_len {
            for kv_h in 0..num_kv_heads {
                // Read the KV head once
                let kv_offset = (seq_idx * num_kv_heads + kv_h) * head_dim;
                let kv_head = &kv[kv_offset..kv_offset + head_dim];

                // Repeat it n_rep times
                for rep in 0..n_rep {
                    let out_h = kv_h * n_rep + rep;
                    let out_offset = (seq_idx * num_kv_heads * n_rep + out_h) * head_dim;
                    output[out_offset..out_offset + head_dim].copy_from_slice(kv_head);
                }
            }
        }

        output
    }

    /// Compute scaled dot-product attention (Candle-style with batched operations)
    ///
    /// Follows Candle's approach:
    /// 1. Repeat K/V for GQA (if num_heads != num_kv_heads)
    /// 2. Compute attention scores: Q @ K^T / sqrt(head_dim)
    /// 3. Apply causal mask
    /// 4. Softmax over scores
    /// 5. Apply attention: scores @ V
    pub fn compute_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<f32>> {
        let head_dim = self.head_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let n_rep = num_heads / num_kv_heads;

        // Q shape: [seq_len, num_heads, head_dim]
        // K, V shape: [kv_seq_len, num_kv_heads, head_dim]
        let kv_seq_len = k.len() / (num_kv_heads * head_dim);

        if std::env::var("DEBUG_ATTN").is_ok() {
            eprintln!(
                "ATTN: seq_len={}, kv_seq_len={}, position={}, n_rep={}",
                seq_len, kv_seq_len, position, n_rep
            );
            eprintln!("  Q shape: [{}x{}x{}]", seq_len, num_heads, head_dim);
            eprintln!("  K/V shape: [{}x{}x{}]", kv_seq_len, num_kv_heads, head_dim);
        }

        // Repeat K/V to match num_heads (for GQA/MQA)
        let k_repeated = if n_rep > 1 { self.repeat_kv(k, kv_seq_len, n_rep) } else { k.to_vec() };
        let v_repeated = if n_rep > 1 { self.repeat_kv(v, kv_seq_len, n_rep) } else { v.to_vec() };

        if std::env::var("DEBUG_ATTN").is_ok() && n_rep > 1 {
            eprintln!("  K/V repeated to shape: [{}x{}x{}]", kv_seq_len, num_heads, head_dim);
        }

        // Compute attention for each head independently
        let mut output = vec![0.0; seq_len * num_heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..num_heads {
            for i in 0..seq_len {
                // Get query vector: Q[i, h, :]
                let q_offset = (i * num_heads + h) * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                // Compute scores = Q[i,h] @ K[j,h]^T for all j
                let mut scores = vec![0.0; kv_seq_len];
                for (j, score_ref) in scores.iter_mut().enumerate().take(kv_seq_len) {
                    let k_offset = (j * num_heads + h) * head_dim;
                    let k_vec = &k_repeated[k_offset..k_offset + head_dim];

                    // Dot product: Q @ K^T
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        score += q_vec[d] * k_vec[d];
                    }
                    *score_ref = score * scale;

                    // Causal masking
                    let query_abs_pos = position + i;
                    if j > query_abs_pos {
                        *score_ref = f32::NEG_INFINITY;
                    }
                }

                // Softmax
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0;
                for score in &mut scores {
                    if score.is_finite() {
                        *score = (*score - max_score).exp();
                        exp_sum += *score;
                    } else {
                        *score = 0.0;
                    }
                }
                if exp_sum > 0.0 {
                    for score in &mut scores {
                        *score /= exp_sum;
                    }
                }

                // Debug attention weights
                if std::env::var("DEBUG_ATTN_WEIGHTS").is_ok() && h == 0 && i == 0 {
                    eprintln!(
                        "  Attention weights (head 0, query 0): {:?}",
                        &scores[..kv_seq_len.min(5)]
                    );
                }

                // Weighted sum: output = scores @ V
                let out_offset = (i * num_heads + h) * head_dim;
                for (j, &weight) in scores.iter().enumerate().take(kv_seq_len) {
                    let v_offset = (j * num_heads + h) * head_dim;
                    let v_vec = &v_repeated[v_offset..v_offset + head_dim];

                    for d in 0..head_dim {
                        output[out_offset + d] += weight * v_vec[d];
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Attention weight matrices
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AttentionWeights {
    /// Query projection [hidden_size, hidden_size]
    pub wq: Vec<f32>,
    /// Key projection [hidden_size, num_kv_heads * head_dim]
    pub wk: Vec<f32>,
    /// Value projection [hidden_size, num_kv_heads * head_dim]
    pub wv: Vec<f32>,
    /// Output projection [hidden_size, hidden_size]
    pub wo: Vec<f32>,
}

#[allow(dead_code)]
impl AttentionWeights {
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let kv_size = config.num_kv_heads * (hidden_size / config.num_heads);

        Self {
            wq: vec![0.0; hidden_size * hidden_size],
            wk: vec![0.0; hidden_size * kv_size],
            wv: vec![0.0; hidden_size * kv_size],
            wo: vec![0.0; hidden_size * hidden_size],
        }
    }
}
