//! Transformer architecture implementation
//!
//! Implements the core transformer components for LLM inference.

use wasm_chord_core::error::Result;
use wasm_chord_cpu::matmul_f32;

/// Transformer configuration
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Intermediate (FFN) size
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta
    pub rope_theta: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 4,  // GQA: Grouped Query Attention
            intermediate_size: 5632,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

/// KV cache for a single layer
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached keys [batch, seq_len, num_kv_heads, head_dim]
    pub keys: Vec<f32>,
    /// Cached values [batch, seq_len, num_kv_heads, head_dim]
    pub values: Vec<f32>,
    /// Current sequence position
    pub seq_pos: usize,
    /// Maximum cache size
    pub max_size: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let max_size = max_seq_len * num_kv_heads * head_dim;
        Self {
            keys: vec![0.0; max_size],
            values: vec![0.0; max_size],
            seq_pos: 0,
            max_size,
        }
    }

    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.seq_pos = 0;
    }

    pub fn append(&mut self, keys: &[f32], values: &[f32]) {
        let size = keys.len();
        let start = self.seq_pos * size / self.keys.len() * self.max_size;
        let end = start + size;

        if end <= self.max_size {
            self.keys[start..end].copy_from_slice(keys);
            self.values[start..end].copy_from_slice(values);
            self.seq_pos += 1;
        }
    }
}

/// Multi-head attention layer
pub struct MultiHeadAttention {
    config: TransformerConfig,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(config: TransformerConfig) -> Self {
        let head_dim = config.hidden_size / config.num_heads;
        Self { config, head_dim }
    }

    /// Apply attention with KV caching
    ///
    /// # Arguments
    /// * `hidden_states` - Input [batch, seq_len, hidden_size]
    /// * `weights` - Model weights (wq, wk, wv, wo)
    /// * `kv_cache` - KV cache for this layer
    /// * `position` - Current position in sequence
    ///
    /// # Returns
    /// Output [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &[f32],
        weights: &AttentionWeights,
        kv_cache: &mut KVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.config.hidden_size;
        let hidden_size = self.config.hidden_size;

        // Project to Q, K, V
        let mut q = vec![0.0; seq_len * hidden_size];
        let mut k = vec![0.0; seq_len * self.config.num_kv_heads * self.head_dim];
        let mut v = vec![0.0; seq_len * self.config.num_kv_heads * self.head_dim];

        // Q projection: [seq_len, hidden_size] x [hidden_size, hidden_size]
        matmul_f32(
            hidden_states,
            &weights.wq,
            &mut q,
            seq_len,
            hidden_size,
            hidden_size,
        )?;

        // K projection: [seq_len, hidden_size] x [hidden_size, num_kv_heads * head_dim]
        matmul_f32(
            hidden_states,
            &weights.wk,
            &mut k,
            seq_len,
            hidden_size,
            self.config.num_kv_heads * self.head_dim,
        )?;

        // V projection
        matmul_f32(
            hidden_states,
            &weights.wv,
            &mut v,
            seq_len,
            hidden_size,
            self.config.num_kv_heads * self.head_dim,
        )?;

        // Apply RoPE (Rotary Position Embedding)
        self.apply_rope(&mut q, position)?;
        self.apply_rope(&mut k, position)?;

        // Cache K, V
        kv_cache.append(&k, &v);

        // Compute attention
        let output = self.compute_attention(&q, &kv_cache.keys, &kv_cache.values, seq_len)?;

        // Output projection
        let mut result = vec![0.0; seq_len * hidden_size];
        matmul_f32(&output, &weights.wo, &mut result, seq_len, hidden_size, hidden_size)?;

        Ok(result)
    }

    fn apply_rope(&self, tensor: &mut [f32], position: usize) -> Result<()> {
        // Simplified RoPE implementation
        // TODO: Full RoPE with proper frequency computation
        let head_dim = self.head_dim;
        let num_heads = tensor.len() / head_dim;

        for head in 0..num_heads {
            for i in 0..head_dim / 2 {
                let idx = head * head_dim + i;
                let freq = 1.0 / self.config.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;

                let cos = angle.cos();
                let sin = angle.sin();

                let x0 = tensor[idx];
                let x1 = tensor[idx + head_dim / 2];

                tensor[idx] = x0 * cos - x1 * sin;
                tensor[idx + head_dim / 2] = x0 * sin + x1 * cos;
            }
        }

        Ok(())
    }

    fn compute_attention(
        &self,
        _q: &[f32],
        _k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        // Simplified attention: Q @ K^T @ V
        // TODO: Implement proper scaled dot-product attention with softmax

        let head_dim = self.head_dim;
        let num_heads = self.config.num_heads;

        let mut output = vec![0.0; seq_len * num_heads * head_dim];

        // For now, just copy values (placeholder)
        // Real implementation needs:
        // 1. Reshape Q, K, V to [num_heads, seq_len, head_dim]
        // 2. Compute scores = (Q @ K^T) / sqrt(head_dim)
        // 3. Apply softmax
        // 4. Multiply by V
        // 5. Concatenate heads

        if !v.is_empty() {
            let len = v.len().min(output.len());
            output[..len].copy_from_slice(&v[..len]);
        }

        Ok(output)
    }
}

/// Attention weight matrices
#[derive(Debug, Clone)]
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

/// Feed-forward network layer
pub struct FeedForward {
    config: TransformerConfig,
}

impl FeedForward {
    pub fn new(config: TransformerConfig) -> Self {
        Self { config }
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        weights: &FFNWeights,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.config.hidden_size;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Gate projection
        let mut gate = vec![0.0; seq_len * intermediate_size];
        matmul_f32(hidden_states, &weights.w_gate, &mut gate, seq_len, hidden_size, intermediate_size)?;

        // Up projection
        let mut up = vec![0.0; seq_len * intermediate_size];
        matmul_f32(hidden_states, &weights.w_up, &mut up, seq_len, hidden_size, intermediate_size)?;

        // SiLU activation: gate * sigmoid(gate) * up
        for i in 0..gate.len() {
            let silu = gate[i] / (1.0 + (-gate[i]).exp());
            gate[i] = silu * up[i];
        }

        // Down projection
        let mut output = vec![0.0; seq_len * hidden_size];
        matmul_f32(&gate, &weights.w_down, &mut output, seq_len, intermediate_size, hidden_size)?;

        Ok(output)
    }
}

/// FFN weight matrices
#[derive(Debug, Clone)]
pub struct FFNWeights {
    /// Gate projection [hidden_size, intermediate_size]
    pub w_gate: Vec<f32>,
    /// Up projection [hidden_size, intermediate_size]
    pub w_up: Vec<f32>,
    /// Down projection [intermediate_size, hidden_size]
    pub w_down: Vec<f32>,
}

impl FFNWeights {
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Self {
            w_gate: vec![0.0; hidden_size * intermediate_size],
            w_up: vec![0.0; hidden_size * intermediate_size],
            w_down: vec![0.0; intermediate_size * hidden_size],
        }
    }
}

/// Single transformer layer
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub ffn: FeedForward,
    pub attention_weights: AttentionWeights,
    pub ffn_weights: FFNWeights,
    pub attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
}

impl TransformerLayer {
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            attention: MultiHeadAttention::new(config.clone()),
            ffn: FeedForward::new(config.clone()),
            attention_weights: AttentionWeights::new(config),
            ffn_weights: FFNWeights::new(config),
            attention_norm: vec![1.0; config.hidden_size],
            ffn_norm: vec![1.0; config.hidden_size],
        }
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        kv_cache: &mut KVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        // Pre-norm architecture (like LLaMA)

        // 1. Attention block with residual
        let normed = self.rms_norm(hidden_states, &self.attention_norm)?;
        let attn_output = self.attention.forward(&normed, &self.attention_weights, kv_cache, position)?;

        let mut hidden = hidden_states.to_vec();
        for i in 0..hidden.len() {
            hidden[i] += attn_output[i];
        }

        // 2. FFN block with residual
        let normed = self.rms_norm(&hidden, &self.ffn_norm)?;
        let ffn_output = self.ffn.forward(&normed, &self.ffn_weights)?;

        for i in 0..hidden.len() {
            hidden[i] += ffn_output[i];
        }

        Ok(hidden)
    }

    fn rms_norm(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        let hidden_size = weight.len();
        let seq_len = input.len() / hidden_size;
        let mut output = vec![0.0; input.len()];

        for seq in 0..seq_len {
            let offset = seq * hidden_size;
            let slice = &input[offset..offset + hidden_size];

            // Compute RMS
            let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / hidden_size as f32).sqrt() + 1e-5;

            // Normalize and scale
            for i in 0..hidden_size {
                output[offset + i] = (slice[i] / rms) * weight[i];
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(2048, 4, 64);
        assert_eq!(cache.seq_pos, 0);
        assert_eq!(cache.max_size, 2048 * 4 * 64);
    }

    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 22);
    }

    #[test]
    fn test_attention_weights_creation() {
        let config = TransformerConfig::default();
        let weights = AttentionWeights::new(&config);

        assert_eq!(weights.wq.len(), config.hidden_size * config.hidden_size);
        assert_eq!(weights.wo.len(), config.hidden_size * config.hidden_size);
    }

    #[test]
    fn test_ffn_weights_creation() {
        let config = TransformerConfig::default();
        let weights = FFNWeights::new(&config);

        assert_eq!(weights.w_gate.len(), config.hidden_size * config.intermediate_size);
        assert_eq!(weights.w_down.len(), config.intermediate_size * config.hidden_size);
    }
}
