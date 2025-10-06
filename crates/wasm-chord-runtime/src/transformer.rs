//! Transformer architecture implementation
//!
//! Implements the core transformer components for LLM inference.

use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use wasm_chord_core::error::Result;
use wasm_chord_core::Tokenizer;
use wasm_chord_cpu::matmul_f32;

#[cfg(feature = "gpu")]
use wasm_chord_gpu::GpuBackend;

/// Transformer configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
            num_kv_heads: 4, // GQA: Grouped Query Attention
            intermediate_size: 5632,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

impl From<wasm_chord_core::TransformerConfigData> for TransformerConfig {
    fn from(data: wasm_chord_core::TransformerConfigData) -> Self {
        Self {
            vocab_size: data.vocab_size,
            hidden_size: data.hidden_size,
            num_layers: data.num_layers,
            num_heads: data.num_heads,
            num_kv_heads: data.num_kv_heads,
            intermediate_size: data.intermediate_size,
            max_seq_len: data.max_seq_len,
            rms_norm_eps: data.rms_norm_eps,
            rope_theta: data.rope_theta,
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

#[allow(dead_code)]
impl KVCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let max_size = max_seq_len * num_kv_heads * head_dim;
        Self { keys: vec![0.0; max_size], values: vec![0.0; max_size], seq_pos: 0, max_size }
    }

    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.seq_pos = 0;
    }

    pub fn append(&mut self, keys: &[f32], values: &[f32]) {
        let size = keys.len();
        // Calculate position based on current cache state, not append count
        let start = self.seq_pos;
        let end = start + size;

        if end <= self.max_size {
            self.keys[start..end].copy_from_slice(keys);
            self.values[start..end].copy_from_slice(values);
            // Increment by number of elements added
            // seq_pos tracks the element offset in the flattened cache arrays
            self.seq_pos += size;
        }
    }
}

/// Multi-head attention layer
#[allow(dead_code)]
pub struct MultiHeadAttention {
    config: TransformerConfig,
    head_dim: usize,
}

#[allow(dead_code)]
impl MultiHeadAttention {
    pub fn new(config: TransformerConfig) -> Self {
        let head_dim = config.hidden_size / config.num_heads;
        Self { config, head_dim }
    }

    /// Helper: matrix multiplication with GPU/CPU fallback
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        #[cfg(feature = "gpu")] gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu {
            if let Ok(result) = gpu.matmul(a, b, m as u32, k as u32, n as u32) {
                return Ok(result);
            }
        }

        // CPU fallback
        let mut result = vec![0.0; m * n];
        matmul_f32(a, b, &mut result, m, k, n)?;
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
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        // K projection: [seq_len, hidden_size] x [hidden_size, num_kv_heads * head_dim]
        let k = self.matmul(
            hidden_states,
            &weights.wk,
            seq_len,
            hidden_size,
            self.config.num_kv_heads * self.head_dim,
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
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        // Apply RoPE (Rotary Position Embedding)
        // Apply different positions for each token in the sequence
        let mut q = q;
        let mut k = k;
        self.apply_rope(&mut q, position, seq_len, self.config.num_heads)?;
        self.apply_rope(&mut k, position, seq_len, self.config.num_kv_heads)?;

        // Cache K, V
        if std::env::var("DEBUG_KV").is_ok() {
            eprintln!(
                "  Before append: kv_cache.seq_pos={}, adding {} elements",
                kv_cache.seq_pos,
                k.len()
            );
            eprintln!("    K values being added (first 5): {:?}", &k[..k.len().min(5)]);
        }
        kv_cache.append(&k, &v);
        if std::env::var("DEBUG_KV").is_ok() {
            eprintln!("  After append: kv_cache.seq_pos={}", kv_cache.seq_pos);
            eprintln!("    Cache K values (first 5): {:?}", &kv_cache.keys[..5]);
        }

        // Compute attention (pass position for correct causal masking)
        // Only pass the valid portion of the cache (up to seq_pos)
        let output = self.compute_attention(
            &q,
            &kv_cache.keys[..kv_cache.seq_pos],
            &kv_cache.values[..kv_cache.seq_pos],
            seq_len,
            position,
        )?;

        // Output projection
        let result = self.matmul(
            &output,
            &weights.wo,
            seq_len,
            hidden_size,
            hidden_size,
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

        // For each token in the sequence
        for seq_idx in 0..seq_len {
            let token_pos = start_pos + seq_idx;

            // For each head
            for head in 0..num_heads {
                // For each pair of dimensions
                for i in 0..head_dim / 2 {
                    // Calculate index in tensor: [seq_idx][head][pair_idx]
                    let base_idx = (seq_idx * num_heads + head) * head_dim;
                    let idx0 = base_idx + i;
                    let idx1 = base_idx + i + head_dim / 2;

                    // RoPE frequency calculation
                    let freq = 1.0 / self.config.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                    let angle = token_pos as f32 * freq;

                    let cos = angle.cos();
                    let sin = angle.sin();

                    // Rotate the pair
                    let x0 = tensor[idx0];
                    let x1 = tensor[idx1];

                    tensor[idx0] = x0 * cos - x1 * sin;
                    tensor[idx1] = x0 * sin + x1 * cos;
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

    /// Compute scaled dot-product attention (exposed for benchmarking)
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

        // GQA: num_heads query heads, num_kv_heads key/value heads
        // Each KV head is shared by (num_heads / num_kv_heads) query heads
        let num_queries_per_kv = num_heads / num_kv_heads;

        // Q shape: [seq_len, num_heads, head_dim]
        // K, V shape: [kv_seq_len, num_kv_heads, head_dim]
        let kv_seq_len = k.len() / (num_kv_heads * head_dim);

        if std::env::var("DEBUG_ATTN").is_ok() {
            eprintln!(
                "ATTN: seq_len={}, kv_seq_len={}, position={}",
                seq_len, kv_seq_len, position
            );
            eprintln!("  Q shape: [{}x{}x{}]", seq_len, num_heads, head_dim);
            eprintln!("  K/V shape in cache: [{}x{}x{}]", kv_seq_len, num_kv_heads, head_dim);
        }

        let mut output = vec![0.0; seq_len * num_heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Process each query head
        for h in 0..num_heads {
            // Determine which KV head to use (for GQA)
            let kv_h = h / num_queries_per_kv;

            for i in 0..seq_len {
                // Get query vector for this position and head
                let q_offset = (i * num_heads + h) * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                // Compute attention scores for all KV positions
                let mut scores = vec![0.0; kv_seq_len];

                #[allow(clippy::needless_range_loop)]
                for j in 0..kv_seq_len {
                    // Get key vector
                    let k_offset = (j * num_kv_heads + kv_h) * head_dim;
                    let k_vec = &k[k_offset..k_offset + head_dim];

                    // Compute dot product: Q · K^T (optimized)
                    let score = self.dot_product(q_vec, k_vec);

                    // Scale by sqrt(head_dim)
                    scores[j] = score * scale;

                    // Causal masking: can only attend to positions up to and including current position
                    // llama2.c does: for (t = 0; t <= pos; t++) { compute attention }
                    // So position `position + i` can attend to all j <= position + i
                    let query_abs_pos = position + i;
                    if j > query_abs_pos {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }

                // Softmax over scores
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_scores = vec![0.0; kv_seq_len];
                let mut sum_exp = 0.0;

                for j in 0..kv_seq_len {
                    if scores[j].is_finite() {
                        exp_scores[j] = (scores[j] - max_score).exp();
                        sum_exp += exp_scores[j];
                    }
                }

                // Normalize
                if sum_exp > 0.0 {
                    for score in &mut exp_scores {
                        *score /= sum_exp;
                    }
                }

                // Debug: show attention weights for first head, first query
                if std::env::var("DEBUG_ATTN_WEIGHTS").is_ok() && h == 0 && i == 0 {
                    eprintln!(
                        "  Attention weights (head 0, query 0): {:?}",
                        &exp_scores[..kv_seq_len.min(5)]
                    );
                }

                // Weighted sum of values
                let out_offset = (i * num_heads + h) * head_dim;

                #[allow(clippy::needless_range_loop)]
                for j in 0..kv_seq_len {
                    let v_offset = (j * num_kv_heads + kv_h) * head_dim;
                    let v_vec = &v[v_offset..v_offset + head_dim];
                    let weight = exp_scores[j];

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

/// Feed-forward network layer
#[allow(dead_code)]
pub struct FeedForward {
    config: TransformerConfig,
}

#[allow(dead_code)]
impl FeedForward {
    pub fn new(config: TransformerConfig) -> Self {
        Self { config }
    }

    /// Helper: matrix multiplication with GPU/CPU fallback
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        #[cfg(feature = "gpu")] gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu {
            if let Ok(result) = gpu.matmul(a, b, m as u32, k as u32, n as u32) {
                return Ok(result);
            }
        }

        // CPU fallback
        let mut result = vec![0.0; m * n];
        matmul_f32(a, b, &mut result, m, k, n)?;
        Ok(result)
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        weights: &FFNWeights,
        #[cfg(feature = "gpu")] gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.config.hidden_size;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Gate projection
        let mut gate = self.matmul(
            hidden_states,
            &weights.w_gate,
            seq_len,
            hidden_size,
            intermediate_size,
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        // Up projection
        let up = self.matmul(
            hidden_states,
            &weights.w_up,
            seq_len,
            hidden_size,
            intermediate_size,
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        // SiLU activation: gate * sigmoid(gate) * up
        for i in 0..gate.len() {
            let silu = gate[i] / (1.0 + (-gate[i]).exp());
            gate[i] = silu * up[i];
        }

        // Down projection
        let output = self.matmul(
            &gate,
            &weights.w_down,
            seq_len,
            intermediate_size,
            hidden_size,
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        Ok(output)
    }
}

/// FFN weight matrices
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FFNWeights {
    /// Gate projection [hidden_size, intermediate_size]
    pub w_gate: Vec<f32>,
    /// Up projection [hidden_size, intermediate_size]
    pub w_up: Vec<f32>,
    /// Down projection [intermediate_size, hidden_size]
    pub w_down: Vec<f32>,
}

#[allow(dead_code)]
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
#[allow(dead_code)]
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub ffn: FeedForward,
    pub attention_weights: AttentionWeights,
    pub ffn_weights: FFNWeights,
    pub attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
}

#[allow(dead_code)]
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
        #[cfg(feature = "gpu")] gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        // Pre-norm architecture (like LLaMA)

        // 1. Attention block with residual
        let normed = self.rms_norm(hidden_states, &self.attention_norm)?;
        let attn_output = self.attention.forward(
            &normed,
            &self.attention_weights,
            kv_cache,
            position,
            #[cfg(feature = "gpu")]
            gpu,
        )?;

        let mut hidden = hidden_states.to_vec();
        for i in 0..hidden.len() {
            hidden[i] += attn_output[i];
        }

        // 2. FFN block with residual
        let normed = self.rms_norm(&hidden, &self.ffn_norm)?;
        let ffn_output = self.ffn.forward(
            &normed,
            &self.ffn_weights,
            #[cfg(feature = "gpu")]
            gpu,
        )?;

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

/// Generation options
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_tokens: 100, temperature: 0.7, top_p: 1.0, top_k: 0, repetition_penalty: 1.1 }
    }
}

/// Complete transformer model with embeddings and output head
pub struct Model {
    /// Model configuration
    pub config: TransformerConfig,
    /// Token embeddings [vocab_size, hidden_size]
    pub token_embeddings: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<TransformerLayer>,
    /// Output normalization
    pub output_norm: Vec<f32>,
    /// LM head (language model head) [hidden_size, vocab_size]
    /// Often tied with token_embeddings (weight sharing)
    pub lm_head: Vec<f32>,
    /// KV caches for each layer
    pub kv_caches: Vec<KVCache>,
    /// GPU backend (optional, enabled with "gpu" feature)
    #[cfg(feature = "gpu")]
    gpu: Option<GpuBackend>,
}

/// Transpose a matrix stored in row-major order
/// Input: [rows, cols] in row-major order
/// Output: [cols, rows] in row-major order
fn transpose_matrix(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
    transposed
}

impl Model {
    /// Create a new model with initialized (zero) weights
    pub fn new(config: TransformerConfig) -> Self {
        let mut layers = Vec::new();
        let mut kv_caches = Vec::new();
        let head_dim = config.hidden_size / config.num_heads;

        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(&config));
            kv_caches.push(KVCache::new(config.max_seq_len, config.num_kv_heads, head_dim));
        }

        Self {
            token_embeddings: vec![0.0; config.vocab_size * config.hidden_size],
            output_norm: vec![1.0; config.hidden_size],
            lm_head: vec![0.0; config.hidden_size * config.vocab_size],
            kv_caches,
            layers,
            config,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Initialize GPU backend (if feature enabled)
    #[cfg(feature = "gpu")]
    pub fn init_gpu(&mut self) -> Result<()> {
        if GpuBackend::is_available() {
            match pollster::block_on(GpuBackend::new()) {
                Ok(gpu) => {
                    eprintln!("✅ GPU backend initialized successfully");
                    self.gpu = Some(gpu);
                    Ok(())
                }
                Err(e) => {
                    eprintln!("⚠️  GPU initialization failed: {}", e);
                    Ok(()) // Fall back to CPU
                }
            }
        } else {
            eprintln!("⚠️  GPU not available, using CPU backend");
            Ok(())
        }
    }

    /// Matrix multiplication with GPU/CPU fallback
    ///
    /// Tries GPU first (if enabled), falls back to CPU
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu {
            // Try GPU matmul
            match gpu.matmul(a, b, m as u32, k as u32, n as u32) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("⚠️  GPU matmul failed: {}, falling back to CPU", e);
                }
            }
        }

        // CPU fallback
        let mut result = vec![0.0; m * n];
        matmul_f32(a, b, &mut result, m, k, n)?;
        Ok(result)
    }

    /// Load weights from GGUF tensors
    pub fn load_from_gguf<R: std::io::Read + std::io::Seek>(
        &mut self,
        tensor_loader: &mut wasm_chord_core::tensor_loader::TensorLoader,
        parser: &mut wasm_chord_core::formats::gguf::GGUFParser<R>,
    ) -> Result<()> {
        use wasm_chord_core::error::Error;

        // Load token embeddings
        if let Ok(embedding_data) = tensor_loader.load_tensor("token_embd.weight", parser) {
            let has_nan = embedding_data.iter().any(|&x| x.is_nan());
            let has_inf = embedding_data.iter().any(|&x| x.is_infinite());
            let sum: f32 = embedding_data.iter().take(100).sum();
            let min = embedding_data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = embedding_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("Loaded token_embd.weight: {} elements, nan={}, inf={}, sum100={:.6}, range=[{:.6}, {:.6}]",
                      embedding_data.len(), has_nan, has_inf, sum, min, max);
            self.token_embeddings.copy_from_slice(embedding_data);
        } else {
            return Err(Error::ParseError("Missing token embeddings".to_string()));
        }

        // Load output norm
        if let Ok(norm_data) = tensor_loader.load_tensor("output_norm.weight", parser) {
            self.output_norm.copy_from_slice(norm_data);
        }

        // Load LM head (or tie with embeddings)
        if let Ok(lm_head_data) = tensor_loader.load_tensor("output.weight", parser) {
            // GGUF has [hidden_size, vocab_size], transpose to [vocab_size, hidden_size] for our use
            // Actually wait - we WANT [hidden_size, vocab_size], so no transpose needed!
            self.lm_head.copy_from_slice(lm_head_data);
        } else {
            // Weight tying: LM head shares weights with token embeddings
            // Token embeddings are [vocab_size, hidden_size], transpose to [hidden_size, vocab_size]
            let lm_head_transposed = transpose_matrix(
                &self.token_embeddings,
                self.config.vocab_size,
                self.config.hidden_size,
            );
            self.lm_head.copy_from_slice(&lm_head_transposed);
        }

        // Load each layer's weights
        for layer_idx in 0..self.config.num_layers {
            let layer = &mut self.layers[layer_idx];

            // Attention weights
            let wq_name = format!("blk.{}.attn_q.weight", layer_idx);
            let wk_name = format!("blk.{}.attn_k.weight", layer_idx);
            let wv_name = format!("blk.{}.attn_v.weight", layer_idx);
            let wo_name = format!("blk.{}.attn_output.weight", layer_idx);

            // Try without .weight suffix if not found
            let wq_name = if tensor_loader.get_metadata(&wq_name).is_some() {
                wq_name
            } else {
                format!("blk.{}.attn_q", layer_idx)
            };
            let wk_name = if tensor_loader.get_metadata(&wk_name).is_some() {
                wk_name
            } else {
                format!("blk.{}.attn_k", layer_idx)
            };
            let wv_name = if tensor_loader.get_metadata(&wv_name).is_some() {
                wv_name
            } else {
                format!("blk.{}.attn_v", layer_idx)
            };
            let wo_name = if tensor_loader.get_metadata(&wo_name).is_some() {
                wo_name
            } else {
                format!("blk.{}.attn_output", layer_idx)
            };

            if let Ok(wq) = tensor_loader.load_tensor(&wq_name, parser) {
                // GGUF stores as [hidden_size, hidden_size] - use directly
                layer.attention_weights.wq.copy_from_slice(wq);
                if layer_idx == 0 {
                    let has_nan = wq.iter().any(|&x| x.is_nan());
                    let has_inf = wq.iter().any(|&x| x.is_infinite());
                    let sum: f32 = wq.iter().take(100).sum();
                    let min = wq.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = wq.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("Loaded {}: {} elements, nan={}, inf={}, sum100={:.6}, range=[{:.6}, {:.6}]",
                              wq_name, wq.len(), has_nan, has_inf, sum, min, max);
                }
            } else if layer_idx == 0 {
                eprintln!("WARN: Failed to load {}", wq_name);
            }
            if let Ok(wk) = tensor_loader.load_tensor(&wk_name, parser) {
                // GGUF stores as [hidden_size, kv_dim] - use directly
                layer.attention_weights.wk.copy_from_slice(wk);
                if layer_idx == 0 {
                    eprintln!("Loaded {}: {} elements", wk_name, wk.len());
                }
            } else if layer_idx == 0 {
                eprintln!("WARN: Failed to load {}", wk_name);
            }
            if let Ok(wv) = tensor_loader.load_tensor(&wv_name, parser) {
                // GGUF stores as [hidden_size, kv_dim] - use directly
                layer.attention_weights.wv.copy_from_slice(wv);
                if layer_idx == 0 {
                    eprintln!("Loaded {}: {} elements", wv_name, wv.len());
                }
            } else if layer_idx == 0 {
                eprintln!("WARN: Failed to load {}", wv_name);
            }
            if let Ok(wo) = tensor_loader.load_tensor(&wo_name, parser) {
                // GGUF stores as [hidden_size, hidden_size] - use directly
                layer.attention_weights.wo.copy_from_slice(wo);
                if layer_idx == 0 {
                    eprintln!("Loaded {}: {} elements", wo_name, wo.len());
                }
            } else if layer_idx == 0 {
                eprintln!("WARN: Failed to load {}", wo_name);
            }

            // Attention norm
            let attn_norm_name = format!("blk.{}.attn_norm.weight", layer_idx);
            if let Ok(norm) = tensor_loader.load_tensor(&attn_norm_name, parser) {
                layer.attention_norm.copy_from_slice(norm);
            }

            // FFN weights
            let ffn_gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
            let ffn_up_name = format!("blk.{}.ffn_up.weight", layer_idx);
            let ffn_down_name = format!("blk.{}.ffn_down.weight", layer_idx);

            if let Ok(gate) = tensor_loader.load_tensor(&ffn_gate_name, parser) {
                // GGUF stores as [hidden_size, intermediate_size] - use directly
                layer.ffn_weights.w_gate.copy_from_slice(gate);
            }
            if let Ok(up) = tensor_loader.load_tensor(&ffn_up_name, parser) {
                // GGUF stores as [hidden_size, intermediate_size] - use directly
                layer.ffn_weights.w_up.copy_from_slice(up);
            }
            if let Ok(down) = tensor_loader.load_tensor(&ffn_down_name, parser) {
                // GGUF stores as [intermediate_size, hidden_size] - use directly
                layer.ffn_weights.w_down.copy_from_slice(down);
            }

            // FFN norm
            let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
            if let Ok(norm) = tensor_loader.load_tensor(&ffn_norm_name, parser) {
                layer.ffn_norm.copy_from_slice(norm);
            }
        }

        Ok(())
    }

    /// Forward pass through a single layer
    ///
    /// Helper method to avoid borrow checker issues
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden_states: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        let kv_cache = &mut self.kv_caches[layer_idx];
        #[cfg(feature = "gpu")]
        let gpu = self.gpu.as_ref();

        self.layers[layer_idx].forward(
            hidden_states,
            kv_cache,
            position,
            #[cfg(feature = "gpu")]
            gpu,
        )
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs \[seq_len\]
    /// * `position` - Current position in sequence (for KV cache)
    ///
    /// # Returns
    /// Logits \[seq_len, vocab_size\]
    pub fn forward(&mut self, token_ids: &[u32], position: usize) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        let debug = std::env::var("DEBUG_FORWARD").is_ok();
        if debug {
            eprintln!(
                "🔍 Forward: seq_len={}, position={}, tokens={:?}",
                seq_len, position, token_ids
            );
        }

        // 1. Embed tokens
        let mut hidden_states = vec![0.0; seq_len * hidden_size];
        for (i, &token_id) in token_ids.iter().enumerate() {
            let emb_start = (token_id as usize) * hidden_size;
            let emb_end = emb_start + hidden_size;
            let out_start = i * hidden_size;
            let out_end = out_start + hidden_size;

            if emb_end <= self.token_embeddings.len() {
                hidden_states[out_start..out_end]
                    .copy_from_slice(&self.token_embeddings[emb_start..emb_end]);
            }
        }

        if debug {
            let sum: f32 = hidden_states.iter().take(10).sum();
            eprintln!("  Embeddings sum(first 10): {:.6}", sum);
        }

        // 2. Pass through transformer layers
        let profile = std::env::var("PROFILE").is_ok();
        for layer_idx in 0..self.config.num_layers {
            let layer_start = std::time::Instant::now();
            if std::env::var("DEBUG_KV").is_ok() {
                eprintln!(
                    "🔍 Layer {}, pos={}, kv_cache.seq_pos BEFORE layer={}",
                    layer_idx, position, self.kv_caches[layer_idx].seq_pos
                );
            }
            hidden_states = self.forward_layer(layer_idx, &hidden_states, position)?;
            if profile {
                eprintln!("  Layer {} took {:?}", layer_idx, layer_start.elapsed());
            }
            if std::env::var("DEBUG_KV").is_ok() {
                eprintln!(
                    "✅ Layer {}, pos={}, kv_cache.seq_pos AFTER layer={}",
                    layer_idx, position, self.kv_caches[layer_idx].seq_pos
                );
            }
        }

        // 3. Final normalization
        hidden_states = self.rms_norm(&hidden_states, &self.output_norm)?;

        // 4. Project to vocabulary (LM head)
        let vocab_size = self.config.vocab_size;
        let logits =
            self.matmul(&hidden_states, &self.lm_head, seq_len, hidden_size, vocab_size)?;

        Ok(logits)
    }

    /// Apply repetition penalty to logits
    ///
    /// # Arguments
    /// * `logits` - Logits to modify
    /// * `tokens` - Previously generated tokens
    /// * `penalty` - Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition)
    fn apply_repetition_penalty(&self, logits: &mut [f32], tokens: &[u32], penalty: f32) {
        if penalty == 1.0 || tokens.is_empty() {
            return;
        }

        for &token_id in tokens {
            let idx = token_id as usize;
            if idx < logits.len() {
                // If logit is positive, divide by penalty (reduce it)
                // If logit is negative, multiply by penalty (make it more negative)
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    /// Sample next token from logits
    ///
    /// # Arguments
    /// * `logits` - Logits for last position \[vocab_size\]
    /// * `temperature` - Sampling temperature (default 1.0, 0.0 = greedy)
    /// * `top_p` - Nucleus sampling threshold (0.0-1.0, 1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    ///
    /// # Returns
    /// Sampled token ID
    pub fn sample(&self, logits: &[f32], temperature: f32, top_p: f32, top_k: u32) -> Result<u32> {
        let vocab_size = logits.len();

        // Greedy sampling (deterministic)
        if temperature <= 0.0 {
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0));
        }

        // Apply temperature scaling
        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Find max for numerical stability
        let max_logit = scaled_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Softmax
        let exp_logits: Vec<f32> = scaled_logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let mut probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // Create index-probability pairs
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();

        // Sort by probability (descending)
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k filtering
        if top_k > 0 && (top_k as usize) < vocab_size {
            // Zero out probabilities beyond top-k
            for (idx, _) in indexed_probs.iter().skip(top_k as usize) {
                probs[*idx] = 0.0;
            }
        }

        // Apply top-p (nucleus) filtering
        if top_p < 1.0 && top_p > 0.0 {
            let mut cumulative_prob = 0.0;
            let mut cutoff_idx = vocab_size;

            for (i, (_idx, prob)) in indexed_probs.iter().enumerate() {
                cumulative_prob += prob;
                if cumulative_prob >= top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Zero out probabilities beyond nucleus
            for (idx, _) in indexed_probs.iter().skip(cutoff_idx) {
                probs[*idx] = 0.0;
            }
        }

        // Renormalize probabilities
        let sum_filtered: f32 = probs.iter().sum();
        if sum_filtered > 0.0 {
            for p in &mut probs {
                *p /= sum_filtered;
            }
        } else {
            // Fallback: uniform over top-1
            return Ok(indexed_probs[0].0 as u32);
        }

        // Sample from filtered distribution using true randomness
        let mut rng = thread_rng();

        // Convert probabilities to weights for weighted sampling
        // Filter out zero probabilities for efficiency
        let non_zero_probs: Vec<(usize, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(idx, &p)| (idx, p as f64))
            .collect();

        if non_zero_probs.is_empty() {
            // Fallback to greedy if all probs are zero
            return Ok(indexed_probs[0].0 as u32);
        }

        let weights: Vec<f64> = non_zero_probs.iter().map(|(_, w)| *w).collect();
        let indices: Vec<usize> = non_zero_probs.iter().map(|(i, _)| *i).collect();

        let dist = WeightedIndex::new(&weights).map_err(|e| {
            wasm_chord_core::error::Error::Runtime(format!("Weighted sampling failed: {}", e))
        })?;
        let sampled_idx = dist.sample(&mut rng);
        let token_id = indices[sampled_idx] as u32;

        Ok(token_id)
    }

    /// Generate text from a prompt
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `tokenizer` - Tokenizer for encoding/decoding
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated text
    pub fn generate(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        config: &GenerationConfig,
    ) -> Result<String> {
        let max_tokens = config.max_tokens;
        let temperature = config.temperature;
        let top_p = config.top_p;
        let top_k = config.top_k;
        let repetition_penalty = config.repetition_penalty;
        use wasm_chord_core::error::Error;

        // Clear KV cache for new generation
        self.clear_kv_cache();

        // Encode prompt
        let mut tokens = tokenizer.encode(prompt, true)?;

        if tokens.is_empty() {
            return Err(Error::ParseError("Empty token sequence".to_string()));
        }

        let num_prompt_tokens = tokens.len();
        let mut pos = 0;
        let mut token = tokens[0];

        if std::env::var("DEBUG").is_ok() {
            eprintln!("DEBUG: Starting generation with {} prompt tokens", num_prompt_tokens);
        }

        // Main generation loop - handles both prompt processing and token generation
        // This follows the llama2.c pattern exactly
        while pos < num_prompt_tokens + max_tokens - 1 {
            // Forward pass: process current token at current position
            // This adds the token to KV cache at position 'pos'
            if std::env::var("DEBUG").is_ok() {
                eprintln!("DEBUG: pos={}, token={}", pos, token);
            }

            // Check KV cache state before forward
            if std::env::var("DEBUG_KV").is_ok() && !self.kv_caches.is_empty() {
                eprintln!("  KV cache seq_pos={}", self.kv_caches[0].seq_pos);
            }

            let logits = self.forward(&[token], pos)?;

            // Check KV cache state after forward
            if std::env::var("DEBUG_KV").is_ok() && !self.kv_caches.is_empty() {
                eprintln!("  KV cache seq_pos after={}", self.kv_caches[0].seq_pos);
            }

            // Get logits for the last position
            let mut last_logits = logits[(logits.len() - self.config.vocab_size)..].to_vec();

            // Determine next token
            let next;
            if pos < num_prompt_tokens - 1 {
                // Still processing prompt - force next prompt token
                next = tokens[pos + 1];
            } else {
                // Past prompt - sample next token from logits
                // Apply repetition penalty to discourage repeated tokens
                self.apply_repetition_penalty(&mut last_logits, &tokens, repetition_penalty);

                next = self.sample(&last_logits, temperature, top_p, top_k)?;

                // Check for EOS token
                if next == tokenizer.special_tokens().eos_token_id {
                    break;
                }

                // Add generated token to our sequence
                tokens.push(next);
            }

            // Advance position FIRST, then update token for next iteration
            // This is the key pattern from llama2.c
            pos += 1;
            token = next;
        }

        // Decode all tokens to text (skip special tokens)
        let generated_text = tokenizer.decode(&tokens, true)?;

        Ok(generated_text)
    }

    /// Generate text with streaming callback
    ///
    /// Calls the callback for each generated token in real-time.
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `tokenizer` - Tokenizer for encoding/decoding
    /// * `config` - Generation configuration
    /// * `callback` - Function called for each token: (token_id, decoded_text) -> bool (continue?)
    ///
    /// # Returns
    /// Full generated text
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        config: &GenerationConfig,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let max_tokens = config.max_tokens;
        let temperature = config.temperature;
        let top_p = config.top_p;
        let top_k = config.top_k;
        let repetition_penalty = config.repetition_penalty;
        use wasm_chord_core::error::Error;

        // Clear KV cache for new generation
        self.clear_kv_cache();

        // Encode prompt
        let mut tokens = tokenizer.encode(prompt, true)?;

        if tokens.is_empty() {
            return Err(Error::ParseError("Empty token sequence".to_string()));
        }

        let num_prompt_tokens = tokens.len();
        let mut pos = 0;
        let mut token = tokens[0];

        // Main generation loop
        while pos < num_prompt_tokens + max_tokens - 1 {
            let logits = self.forward(&[token], pos)?;

            let mut last_logits = logits[(logits.len() - self.config.vocab_size)..].to_vec();

            let next;
            if pos < num_prompt_tokens - 1 {
                // Still processing prompt
                next = tokens[pos + 1];
            } else {
                // Generate new token
                self.apply_repetition_penalty(&mut last_logits, &tokens, repetition_penalty);
                next = self.sample(&last_logits, temperature, top_p, top_k)?;

                // Check for EOS
                if next == tokenizer.special_tokens().eos_token_id {
                    break;
                }

                tokens.push(next);

                // Decode just this token and call callback
                let token_text = tokenizer.decode(&[next], true)?;

                // Call callback - if it returns false, stop generation
                if !callback(next, &token_text) {
                    break;
                }
            }

            pos += 1;
            token = next;
        }

        // Return full generated text
        let generated_text = tokenizer.decode(&tokens, true)?;
        Ok(generated_text)
    }

    /// Clear all KV caches (for new sequence)
    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
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
            let rms = (sum_sq / hidden_size as f32).sqrt() + self.config.rms_norm_eps;

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

    #[test]
    fn test_model_creation() {
        let config = TransformerConfig::default();
        let model = Model::new(config.clone());

        assert_eq!(model.layers.len(), config.num_layers);
        assert_eq!(model.kv_caches.len(), config.num_layers);
        assert_eq!(model.token_embeddings.len(), config.vocab_size * config.hidden_size);
        assert_eq!(model.lm_head.len(), config.hidden_size * config.vocab_size);
    }

    #[test]
    fn test_model_forward_pass() {
        // Create a tiny config for testing
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let mut model = Model::new(config);

        // Test forward pass with a small sequence
        let input_tokens = vec![1, 2, 3];
        let result = model.forward(&input_tokens, 0);

        assert!(result.is_ok());
        let logits = result.unwrap();
        assert_eq!(logits.len(), input_tokens.len() * model.config.vocab_size);
    }

    #[test]
    fn test_model_sampling_greedy() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let model = Model::new(config);

        // Create test logits (favor token 5)
        let mut logits = vec![0.0; 10];
        logits[5] = 10.0;

        // Greedy sampling (temperature = 0)
        let sampled = model.sample(&logits, 0.0, 1.0, 0).unwrap();
        assert_eq!(sampled, 5);

        // Greedy with temperature = 1.0
        let sampled = model.sample(&logits, 1.0, 1.0, 0).unwrap();
        assert_eq!(sampled, 5);
    }

    #[test]
    fn test_model_sampling_top_k() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let model = Model::new(config);

        // Create logits with known distribution
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 0.3, 0.2, 0.1, 0.0];

        // Top-k = 3 should only consider indices 2, 3, 4
        let sampled = model.sample(&logits, 1.0, 1.0, 3).unwrap();
        assert!((2..=4).contains(&sampled));
    }

    #[test]
    fn test_model_sampling_top_p() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let model = Model::new(config);

        // Create logits with exponential distribution
        let mut logits = vec![0.0; 10];
        logits[0] = 10.0; // Highest probability
        logits[1] = 5.0;
        logits[2] = 2.0;
        logits[3] = 1.0;
        // Rest are much lower

        // Top-p = 0.9 should focus on top few tokens
        let sampled = model.sample(&logits, 1.0, 0.9, 0).unwrap();
        assert!(sampled <= 3, "sampled token should be in nucleus");
    }

    #[test]
    fn test_model_sampling_temperature() {
        let config = TransformerConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let model = Model::new(config);

        let mut logits = vec![1.0; 10];
        logits[5] = 2.0;

        // Low temperature should still prefer token 5
        let sampled = model.sample(&logits, 0.1, 1.0, 0).unwrap();
        assert_eq!(sampled, 5);

        // High temperature makes distribution more uniform
        let sampled = model.sample(&logits, 10.0, 1.0, 0).unwrap();
        // Should still work (may or may not be 5)
        assert!(sampled < 10);
    }

    #[test]
    fn test_attention_computation() {
        // Small config for testing
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let attention = MultiHeadAttention::new(config.clone());

        let seq_len = 3;
        let head_dim = config.hidden_size / config.num_heads; // 16

        // Create test Q, K, V
        let q = vec![0.1; seq_len * config.num_heads * head_dim];
        let k = vec![0.2; seq_len * config.num_kv_heads * head_dim];
        let v = vec![0.3; seq_len * config.num_kv_heads * head_dim];

        let result = attention.compute_attention(&q, &k, &v, seq_len, 0);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Check output shape
        assert_eq!(output.len(), seq_len * config.num_heads * head_dim);

        // Output should not be all zeros (attention was computed)
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn test_attention_causal_masking() {
        // Test that causal masking works correctly
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4, // Standard MHA for simplicity
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let attention = MultiHeadAttention::new(config.clone());

        let seq_len = 4;
        let head_dim = config.hidden_size / config.num_heads;

        // Create distinct Q, K, V values
        let mut q = vec![0.0; seq_len * config.num_heads * head_dim];
        let mut k = vec![0.0; seq_len * config.num_kv_heads * head_dim];
        let mut v = vec![0.0; seq_len * config.num_kv_heads * head_dim];

        // Set distinct values for each position
        for i in 0..seq_len {
            for h in 0..config.num_heads {
                for d in 0..head_dim {
                    q[(i * config.num_heads + h) * head_dim + d] = (i + 1) as f32;
                }
            }
            for h in 0..config.num_kv_heads {
                for d in 0..head_dim {
                    k[(i * config.num_kv_heads + h) * head_dim + d] = (i + 1) as f32;
                    v[(i * config.num_kv_heads + h) * head_dim + d] = (i + 1) as f32 * 10.0;
                }
            }
        }

        let result = attention.compute_attention(&q, &k, &v, seq_len, 0).unwrap();

        // Check that output shape is correct
        assert_eq!(result.len(), seq_len * config.num_heads * head_dim);

        // Verify values are reasonable (not NaN, not zero)
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_attention_with_gqa() {
        // Test Grouped Query Attention
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4 query heads per KV head
            intermediate_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let attention = MultiHeadAttention::new(config.clone());

        let seq_len = 2;
        let head_dim = config.hidden_size / config.num_heads; // 8

        let q = vec![1.0; seq_len * config.num_heads * head_dim];
        let k = vec![0.5; seq_len * config.num_kv_heads * head_dim];
        let v = vec![2.0; seq_len * config.num_kv_heads * head_dim];

        let result = attention.compute_attention(&q, &k, &v, seq_len, 0);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Verify correct output shape
        assert_eq!(output.len(), seq_len * config.num_heads * head_dim);

        // All values should be finite
        for &val in &output {
            assert!(val.is_finite());
            assert!(val >= 0.0); // Since V is all positive and weights are normalized
        }
    }
}
