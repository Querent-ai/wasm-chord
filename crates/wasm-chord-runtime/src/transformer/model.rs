//! Complete transformer model implementation

use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use wasm_chord_core::error::Result;
use wasm_chord_core::Tokenizer;
use wasm_chord_cpu::{matmul_f32, matmul_transposed};

#[cfg(feature = "gpu")]
use wasm_chord_gpu::GpuBackend;

use super::{TransformerConfig, GenerationConfig, KVCache, TransformerLayer};

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

/// Compute basic stats for a slice of f32s
fn tensor_stats(name: &str, data: &[f32]) {
    let len = data.len();
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut zero_count = 0usize;

    for &v in data {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        if v == 0.0 {
            zero_count += 1;
        }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
    }

    let mean = if len > 0 { sum / (len as f64) } else { 0.0 };
    let variance = if len > 0 { (sum_sq / (len as f64)) - (mean * mean) } else { 0.0 };
    let std = variance.max(0.0).sqrt();

    eprintln!(
        "STATS '{}' len={} nan={} inf={} zeros={} mean={:.6} std={:.6} min={:.6} max={:.6}",
        name,
        len,
        nan_count,
        inf_count,
        zero_count,
        mean,
        std,
        data.iter().copied().fold(f32::INFINITY, f32::min),
        data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Print a small preview
    let preview: Vec<String> = data.iter().take(8).map(|v| format!("{:.6}", v)).collect();
    eprintln!("  preview: {:?}", preview);
}

/// Naive matmul reference (row-major)
fn matmul_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            r[i * n + j] = sum;
        }
    }
    r
}

/// Test matmul implementation for small sizes and random values
fn matmul_self_test() -> Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let tests = vec![(2, 3, 4), (4, 4, 4), (3, 5, 2)];
    for (m, k, n) in tests {
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for v in &mut a {
            *v = rng.gen_range(-1.0..1.0);
        }
        for v in &mut b {
            *v = rng.gen_range(-1.0..1.0);
        }

        let mut out = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut out, m, k, n)?;

        let expected = matmul_naive(&a, &b, m, k, n);

        // compute max absolute diff
        let mut max_diff = 0.0f32;
        for idx in 0..(m * n) {
            let d = (out[idx] - expected[idx]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("MATMUL TEST {}x{}x{} max_diff={:.6}", m, k, n, max_diff);
        if max_diff > 1e-3 {
            // tolerance for float rounding
            return Err(wasm_chord_core::error::Error::Runtime(format!(
                "Matmul mismatch max_diff={:.6}",
                max_diff
            )));
        }
    }
    Ok(())
}

// Helper function to manually compute a single logit for verification
fn compute_manual_logit(
    hidden_states: &[f32],
    lm_head: &[f32],
    token_id: usize,
    hidden_size: usize,
) -> f32 {
    // The matmul treats hidden_states as [seq_len, hidden_size] where seq_len=1
    // and lm_head as [hidden_size, vocab_size]
    // So for the first (and only) sequence position, we compute:
    // logit = sum(hidden_states[i] * lm_head[i * vocab_size + token_id]) for i in 0..hidden_size

    let vocab_size = 32000;
    let mut sum = 0.0f32;
    for (i, hidden_state) in hidden_states.iter().enumerate().take(hidden_size) {
        let weight_idx = i * vocab_size + token_id;
        if weight_idx < lm_head.len() {
            sum += hidden_state * lm_head[weight_idx];
        }
    }
    sum
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
    ///
    /// # Arguments
    /// * `transposed_b` - If true, B is stored as [n, k] (GGUF format) and will use efficient transpose matmul
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize, transposed_b: bool) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu {
            // Try GPU matmul
            // TODO: GPU also needs to handle transposed case
            if !transposed_b {
                match gpu.matmul(a, b, m as u32, k as u32, n as u32) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        eprintln!("⚠️  GPU matmul failed: {}, falling back to CPU", e);
                    }
                }
            }
        }

        // CPU fallback
        let mut result = vec![0.0; m * n];
        if transposed_b {
            // B is stored as [n, k] (GGUF format), use optimized transpose matmul
            matmul_transposed(a, b, &mut result, m, k, n)?;
        } else {
            // Standard matmul: B is [k, n]
            matmul_f32(a, b, &mut result, m, k, n)?;
        }
        Ok(result)
    }

    /// Load weights from GGUF tensors
    pub fn load_from_gguf<R: std::io::Read + std::io::Seek>(
        &mut self,
        tensor_loader: &mut wasm_chord_core::tensor_loader::TensorLoader,
        parser: &mut wasm_chord_core::formats::gguf::GGUFParser<R>,
    ) -> Result<()> {
        use wasm_chord_core::error::Error;

        // Run matmul self-test first
        eprintln!("🧪 Running matmul self-test...");
        matmul_self_test()?;
        eprintln!("✅ Matmul self-test passed!");

        // Load token embeddings - try different tensor names
        let embedding_data = if let Ok(data) =
            tensor_loader.load_tensor("token_embd.weight", parser)
        {
            println!("✅ Found 'token_embd.weight'");
            data
        } else if let Ok(data) = tensor_loader.load_tensor("model.embed_tokens.weight", parser) {
            println!("✅ Found 'model.embed_tokens.weight'");
            data
        } else if let Ok(data) = tensor_loader.load_tensor("embed_tokens.weight", parser) {
            println!("✅ Found 'embed_tokens.weight'");
            data
        } else {
            return Err(Error::ParseError("Missing token embeddings".to_string()));
        };

        {
            eprintln!("LOADED 'token_embd.weight' raw.len={}", embedding_data.len());
            tensor_stats("token_embd.weight (raw)", embedding_data);
            eprintln!(
                "  Raw preview (first 20): {:?}",
                &embedding_data[..20.min(embedding_data.len())]
            );

            println!(
                "🔍 Loading token embeddings as-is: [vocab_size={}, hidden_size={}]",
                self.config.vocab_size, self.config.hidden_size
            );
            // GGUF stores token_embd.weight as [vocab_size, hidden_size]
            // Our embedding lookup (line 1342) correctly handles this format
            // by gathering row token_id from the matrix
            self.token_embeddings.copy_from_slice(embedding_data);
            eprintln!(
                "  Embeddings preview (first 20): {:?}",
                &self.token_embeddings[..20.min(self.token_embeddings.len())]
            );
            tensor_stats("token_embd.weight (model)", &self.token_embeddings);
        }

        // Load output norm
        if let Ok(norm_data) = tensor_loader.load_tensor("output_norm.weight", parser) {
            eprintln!("LOADED 'output_norm.weight' raw.len={}", norm_data.len());
            tensor_stats("output_norm.weight (raw)", norm_data);

            // Load raw weights without arbitrary scaling
            self.output_norm.copy_from_slice(norm_data);

            tensor_stats("output_norm.weight (model)", &self.output_norm);
            println!("✅ Output norm loaded");
        } else {
            eprintln!("WARN: Failed to load output_norm.weight");
        }

        // Load LM head - try different tensor names
        if let Ok(lm_head_data) = tensor_loader.load_tensor("output.weight", parser) {
            println!("✅ Found 'output.weight'");
            eprintln!("LOADED 'output.weight' raw.len={}", lm_head_data.len());
            tensor_stats("output.weight (raw)", lm_head_data);

            println!("🔍 Loading LM head tensor: {} elements", lm_head_data.len());
            // GGUF stores output.weight as [vocab_size, hidden_size]
            // We need to transpose it to [hidden_size, vocab_size] for matmul
            let lm_head_transposed = transpose_matrix(
                lm_head_data,
                self.config.vocab_size,
                self.config.hidden_size,
            );
            self.lm_head.copy_from_slice(&lm_head_transposed);
            tensor_stats("output.weight (model)", &self.lm_head);
            println!(
                "✅ LM head loaded (shape: [hidden_size={}, vocab_size={}])",
                self.config.hidden_size, self.config.vocab_size
            );
        } else if let Ok(lm_head_data) = tensor_loader.load_tensor("lm_head.weight", parser) {
            println!("✅ Found 'lm_head.weight'");
            eprintln!("LOADED 'lm_head.weight' raw.len={}", lm_head_data.len());
            tensor_stats("lm_head.weight (raw)", lm_head_data);
            // GGUF stores lm_head.weight as [vocab_size, hidden_size]
            // We need to transpose it to [hidden_size, vocab_size] for matmul
            let lm_head_transposed = transpose_matrix(
                lm_head_data,
                self.config.vocab_size,
                self.config.hidden_size,
            );
            self.lm_head.copy_from_slice(&lm_head_transposed);
            tensor_stats("lm_head.weight (model)", &self.lm_head);
            println!("✅ LM head loaded");
        } else if let Ok(lm_head_data) = tensor_loader.load_tensor("model.lm_head.weight", parser) {
            println!("✅ Found 'model.lm_head.weight'");
            eprintln!("LOADED 'model.lm_head.weight' raw.len={}", lm_head_data.len());
            tensor_stats("model.lm_head.weight (raw)", lm_head_data);
            // GGUF stores model.lm_head.weight as [vocab_size, hidden_size]
            // We need to transpose it to [hidden_size, vocab_size] for matmul
            let lm_head_transposed = transpose_matrix(
                lm_head_data,
                self.config.vocab_size,
                self.config.hidden_size,
            );
            self.lm_head.copy_from_slice(&lm_head_transposed);
            tensor_stats("model.lm_head.weight (model)", &self.lm_head);
            println!("✅ LM head loaded");
        } else {
            // Weight tying: LM head shares weights with token embeddings
            println!("🔍 Using weight tying - LM head from token embeddings");
            let lm_head_transposed = transpose_matrix(
                &self.token_embeddings,
                self.config.hidden_size,
                self.config.vocab_size,
            );
            self.lm_head.copy_from_slice(&lm_head_transposed);
            tensor_stats("lm_head (from embeddings, transposed)", &self.lm_head);
            println!("✅ LM head loaded from token embeddings (weight tying)");
        }

        // Print LM head shape check
        eprintln!(
            "LM_HEAD len={}, expected={} (vocab_size * hidden_size={})",
            self.lm_head.len(),
            self.config.vocab_size * self.config.hidden_size,
            self.config.vocab_size * self.config.hidden_size
        );
        eprintln!("token_embeddings.len()={}", self.token_embeddings.len());

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
                // Use GGUF weights directly - no transpose needed
                // matmul_transposed will handle the orientation efficiently
                layer.attention_weights.wq.copy_from_slice(wq);
                if layer_idx == 0 {
                    eprintln!("LOADED '{}' raw.len={}", wq_name, wq.len());
                    tensor_stats(&format!("{} (model)", wq_name), &layer.attention_weights.wq);
                }
            } else if layer_idx == 0 {
                eprintln!("WARN: Failed to load {}", wq_name);
            }
            if let Ok(wk) = tensor_loader.load_tensor(&wk_name, parser) {
                // Use GGUF weights directly - no transpose needed
                // matmul_transposed will handle the orientation efficiently
                layer.attention_weights.wk.copy_from_slice(wk);
                if layer_idx == 0 {
                    eprintln!("LOADED '{}' raw.len={}", wk_name, wk.len());
                    tensor_stats(&format!("{} (model)", wk_name), &layer.attention_weights.wk);
                }
            } else if layer_idx == 0 {
                eprintln!("WARN: Failed to load {}", wk_name);
            }
            if let Ok(wv) = tensor_loader.load_tensor(&wv_name, parser) {
                // Use GGUF weights directly - no transpose needed
                // matmul_transposed will handle the orientation efficiently
                layer.attention_weights.wv.copy_from_slice(wv);
                if layer_idx == 0 {
                    eprintln!("LOADED '{}' raw.len={}", wv_name, wv.len());
                    tensor_stats(&format!("{} (model)", wv_name), &layer.attention_weights.wv);
                }
            } else if layer_idx == 0 {
                eprintln!("WARN: Failed to load {}", wv_name);
            }
            if let Ok(wo) = tensor_loader.load_tensor(&wo_name, parser) {
                if layer_idx == 0 {
                    eprintln!("LOADED '{}' raw.len={}", wo_name, wo.len());
                    tensor_stats(&format!("{} (raw)", wo_name), wo);
                }
                layer.attention_weights.wo.copy_from_slice(wo);
                if layer_idx == 0 {
                    tensor_stats(&format!("{} (model)", wo_name), &layer.attention_weights.wo);
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
                layer.ffn_weights.w_gate.copy_from_slice(gate);
            }
            if let Ok(up) = tensor_loader.load_tensor(&ffn_up_name, parser) {
                layer.ffn_weights.w_up.copy_from_slice(up);
            }
            if let Ok(down) = tensor_loader.load_tensor(&ffn_down_name, parser) {
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
        // GGUF stores embeddings as [vocab_size, hidden_size]
        // So for token_id, we need to gather row token_id from the matrix
        let mut hidden_states = vec![0.0; seq_len * hidden_size];

        for (seq_idx, &token_id) in token_ids.iter().enumerate() {
            let out_start = seq_idx * hidden_size;

            // Gather embedding for this token from the [vocab_size, hidden_size] matrix
            // Element [row, col] is at index row * hidden_size + col
            // So embedding dimension i for token_id is at: token_id * hidden_size + i
            for dim_idx in 0..hidden_size {
                let emb_idx = (token_id as usize) * hidden_size + dim_idx;
                if emb_idx < self.token_embeddings.len() {
                    hidden_states[out_start + dim_idx] = self.token_embeddings[emb_idx];
                } else {
                    eprintln!("WARN: Token {} dim {} out of bounds", token_id, dim_idx);
                    break;
                }
            }
        }

        if debug {
            let sum: f32 = hidden_states.iter().take(10).sum();
            let mean: f32 = hidden_states.iter().sum::<f32>() / hidden_states.len() as f32;
            let max: f32 = hidden_states.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min: f32 = hidden_states.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!("  Embeddings sum(first 10): {:.6}, mean: {:.6}, min: {:.6}, max: {:.6}", sum, mean, min, max);
        }

        // 2. Pass through transformer layers
        let profile = std::env::var("PROFILE").is_ok();
        for layer_idx in 0..self.config.num_layers {
            let layer_start = std::time::Instant::now();
            if std::env::var("DEBUG_KV").is_ok() {
                eprintln!(
                    "🔍 Layer {}, pos={}, kv_cache.current_seq_len BEFORE layer={}",
                    layer_idx, position, self.kv_caches[layer_idx].current_seq_len
                );
            }
            hidden_states = self.forward_layer(layer_idx, &hidden_states, position)?;
            if profile {
                eprintln!("  Layer {} took {:?}", layer_idx, layer_start.elapsed());
            }
            if std::env::var("DEBUG_KV").is_ok() {
                eprintln!(
                    "✅ Layer {}, pos={}, kv_cache.current_seq_len AFTER layer={}",
                    layer_idx, position, self.kv_caches[layer_idx].current_seq_len
                );
            }
        }

        // 3. Final normalization
        hidden_states = self.rms_norm(&hidden_states, &self.output_norm)?;

        // DEBUG: Validate hidden states after final RMSNorm
        if debug {
            let sum: f32 = hidden_states.iter().sum::<f32>();
            let mean: f32 = sum / hidden_states.len() as f32;
            let max: f32 = hidden_states.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min: f32 = hidden_states.iter().copied().fold(f32::INFINITY, f32::min);
            let abs_max: f32 = hidden_states.iter().copied().map(f32::abs).fold(f32::NEG_INFINITY, f32::max);
            eprintln!("  Final hidden states: mean: {:.6}, min: {:.6}, max: {:.6}, abs_max: {:.6}", mean, min, max, abs_max);

            // Check for reasonable ranges
            if abs_max > 10.0 {
                eprintln!("  ⚠️  WARNING: Hidden states have large values (abs_max={:.6})", abs_max);
            }
            if mean.abs() > 1.0 {
                eprintln!("  ⚠️  WARNING: Hidden states have large mean (mean={:.6})", mean);
            }
        }

        // 4. Project to vocabulary (LM head)
        let vocab_size = self.config.vocab_size;

        // DEBUG: Check hidden states before LM head
        if std::env::var("DEBUG_LOGITS").is_ok() {
            eprintln!("🔍 HIDDEN STATES BEFORE LM HEAD:");
            eprintln!("  hidden_states.len() = {}", hidden_states.len());
            eprintln!(
                "  seq_len = {}, hidden_size = {}, vocab_size = {}",
                seq_len, hidden_size, vocab_size
            );

            // Check first few values
            let preview_len = 10.min(hidden_states.len());
            eprintln!("  hidden_states preview: {:?}", &hidden_states[..preview_len]);

            // Check statistics
            let sum: f32 = hidden_states.iter().sum();
            let mean = sum / hidden_states.len() as f32;
            let max = hidden_states.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = hidden_states.iter().copied().fold(f32::INFINITY, f32::min);
            let nan_count = hidden_states.iter().filter(|&&x| x.is_nan()).count();
            let inf_count = hidden_states.iter().filter(|&&x| x.is_infinite()).count();

            eprintln!("  hidden_states stats: sum={:.6}, mean={:.6}, min={:.6}, max={:.6}, nan={}, inf={}",
                     sum, mean, min, max, nan_count, inf_count);

            // Check LM head weights
            eprintln!("🔍 LM HEAD WEIGHTS:");
            eprintln!("  lm_head.len() = {}", self.lm_head.len());
            let lm_preview_len = 10.min(self.lm_head.len());
            eprintln!("  lm_head preview: {:?}", &self.lm_head[..lm_preview_len]);

            let lm_sum: f32 = self.lm_head.iter().sum();
            let lm_mean = lm_sum / self.lm_head.len() as f32;
            let lm_max = self.lm_head.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lm_min = self.lm_head.iter().copied().fold(f32::INFINITY, f32::min);
            let lm_nan_count = self.lm_head.iter().filter(|&&x| x.is_nan()).count();
            let lm_inf_count = self.lm_head.iter().filter(|&&x| x.is_infinite()).count();

            eprintln!(
                "  lm_head stats: sum={:.6}, mean={:.6}, min={:.6}, max={:.6}, nan={}, inf={}",
                lm_sum, lm_mean, lm_min, lm_max, lm_nan_count, lm_inf_count
            );
        }

        let logits =
            self.matmul(&hidden_states, &self.lm_head, seq_len, hidden_size, vocab_size, false)?;

        // DEBUG: Validate logits
        if debug {
            let sum: f32 = logits.iter().sum::<f32>();
            let mean: f32 = sum / logits.len() as f32;
            let max: f32 = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min: f32 = logits.iter().copied().fold(f32::INFINITY, f32::min);
            let abs_max: f32 = logits.iter().copied().map(f32::abs).fold(f32::NEG_INFINITY, f32::max);
            let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
            let inf_count = logits.iter().filter(|&&x| x.is_infinite()).count();

            eprintln!("  Logits: mean: {:.6}, min: {:.6}, max: {:.6}, abs_max: {:.6}, nan: {}, inf: {}",
                     mean, min, max, abs_max, nan_count, inf_count);

            // Check for reasonable ranges
            if abs_max > 20.0 {
                eprintln!("  ⚠️  WARNING: Logits have very large values (abs_max={:.6})", abs_max);
            }
            if mean.abs() > 5.0 {
                eprintln!("  ⚠️  WARNING: Logits have large mean (mean={:.6})", mean);
            }
            if nan_count > 0 {
                eprintln!("  ❌ ERROR: Logits contain NaN values!");
            }
            if inf_count > 0 {
                eprintln!("  ❌ ERROR: Logits contain infinite values!");
            }
        }

        // DEBUG: Check logits after matmul
        if std::env::var("DEBUG_LOGITS").is_ok() {
            eprintln!("🔍 LOGITS AFTER MATMUL:");
            eprintln!("  logits.len() = {}", logits.len());
            eprintln!(
                "  expected len = seq_len * vocab_size = {} * {} = {}",
                seq_len,
                vocab_size,
                seq_len * vocab_size
            );

            // Check first few logits
            let preview_len = 10.min(logits.len());
            eprintln!("  logits preview: {:?}", &logits[..preview_len]);

            // Check statistics
            let logits_sum: f32 = logits.iter().sum();
            let logits_mean = logits_sum / logits.len() as f32;
            let logits_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let logits_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
            let logits_nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
            let logits_inf_count = logits.iter().filter(|&&x| x.is_infinite()).count();

            eprintln!(
                "  logits stats: sum={:.6}, mean={:.6}, min={:.6}, max={:.6}, nan={}, inf={}",
                logits_sum, logits_mean, logits_min, logits_max, logits_nan_count, logits_inf_count
            );

            // Check if the extreme dominance is in the raw logits
            let mut sorted_logits: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            sorted_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            eprintln!("  Top 5 raw logits:");
            for (i, (token_id, logit)) in sorted_logits.iter().take(5).enumerate() {
                eprintln!("    {}: token {} = {:.6}", i + 1, token_id, logit);
            }

            // Check the difference between top 2
            if sorted_logits.len() >= 2 {
                let diff = sorted_logits[0].1 - sorted_logits[1].1;
                eprintln!("  Top 2 difference: {:.6}", diff);
                if diff > 1.0 {
                    eprintln!("  ⚠️  Large difference between top 2 logits: {:.6}", diff);
                }
            }

            // MANUAL VERIFICATION: Compute a few logits manually to check matmul
            eprintln!("🔍 MANUAL MATMUL VERIFICATION:");
            let top_token_id = sorted_logits[0].0;
            let manual_logit =
                compute_manual_logit(&hidden_states, &self.lm_head, top_token_id, hidden_size);
            let matmul_logit = logits[top_token_id];
            eprintln!(
                "  Token {}: matmul={:.6}, manual={:.6}, diff={:.6}",
                top_token_id,
                matmul_logit,
                manual_logit,
                (matmul_logit - manual_logit).abs()
            );

            if (matmul_logit - manual_logit).abs() > 1e-5 {
                eprintln!(
                    "  ❌ MATMUL BUG DETECTED! Manual computation differs from matmul result"
                );
            } else {
                eprintln!("  ✅ Matmul computation verified");
            }
        }

        // Debug: print detailed intermediate values
        if std::env::var("DEBUG_INTERMEDIATE").is_ok() {
            let last_logits = &logits[(logits.len() - vocab_size)..];
            let mut indexed_logits: Vec<(usize, f32)> =
                last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_logits
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            eprintln!("=== INTERMEDIATE VALUES ===");
            eprintln!("  Hidden states sum: {:.6}", hidden_states.iter().sum::<f32>());
            eprintln!(
                "  Hidden states mean: {:.6}",
                hidden_states.iter().sum::<f32>() / hidden_states.len() as f32
            );
            eprintln!("  Hidden states std: {:.6}", {
                let mean = hidden_states.iter().sum::<f32>() / hidden_states.len() as f32;
                let variance = hidden_states.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / hidden_states.len() as f32;
                variance.sqrt()
            });
            eprintln!("  Logits sum: {:.6}", last_logits.iter().sum::<f32>());
            eprintln!(
                "  Logits mean: {:.6}",
                last_logits.iter().sum::<f32>() / last_logits.len() as f32
            );
            eprintln!(
                "  Logits max: {:.6}",
                last_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );
            eprintln!(
                "  Logits min: {:.6}",
                last_logits.iter().copied().fold(f32::INFINITY, f32::min)
            );
            eprintln!("  Top 10 logits: {:?}", &indexed_logits[..10]);

            // Check specific tokens we care about
            let important_tokens = [278, 1234, 13791]; // "the", "answer", "vertices"
            for &token_id in &important_tokens {
                if token_id < last_logits.len() {
                    eprintln!("  Token {} logit: {:.6}", token_id, last_logits[token_id]);
                }
            }
            eprintln!("========================");
        }

        Ok(logits)
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

            // Debug: show top-k filtering
            eprintln!("🔍 Top-k filtering (k={}):", top_k);
            for (i, (idx, prob)) in indexed_probs.iter().take(top_k as usize).enumerate() {
                eprintln!("    {}: token {} = {:.6}", i, idx, prob);
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

        // Debug: show final sampling
        eprintln!("🎲 Final sampling:");
        eprintln!("    Non-zero probs: {}", non_zero_probs.len());
        eprintln!("    Sampled idx: {} -> token {}", sampled_idx, token_id);
        eprintln!("    Token prob: {:.6}", probs[token_id as usize]);

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

        // Create logits processor with configured sampling strategy
        let mut logits_processor = crate::sampling::LogitsProcessor::with_params(
            42, // seed - could be made configurable
            temperature as f64,
            top_p as f64,
            top_k,
            repetition_penalty,
        );

        // Clear KV cache for new generation
        self.clear_kv_cache();

        // Encode prompt
        let mut tokens = tokenizer.encode(prompt, false)?;

        // DEBUG: Dump tokenization details
        eprintln!("PROMPT_STR = {:?}", prompt);
        eprintln!("TOKENS_ENCODED (len={}): {:?}", tokens.len(), tokens);

        if tokens.is_empty() {
            return Err(Error::ParseError("Empty token sequence".to_string()));
        }

        let num_prompt_tokens = tokens.len();
        let mut pos = 0;
        // Don't initialize token here - we'll get it from tokens[pos] in the loop

        if std::env::var("DEBUG").is_ok() {
            eprintln!("DEBUG: Starting generation with {} prompt tokens", num_prompt_tokens);
        }

        // Main generation loop - handles both prompt processing and token generation
        // This follows the llama2.c pattern exactly
        while pos < num_prompt_tokens + max_tokens - 1 {
            // Get the current token to process
            let token = if pos < num_prompt_tokens {
                tokens[pos] // Process prompt token
            } else {
                tokens[tokens.len() - 1] // Process last generated token
            };

            // DEBUG: Dump token processing details
            eprintln!("TOKENS (len={}): {:?}", tokens.len(), tokens);
            if pos < num_prompt_tokens {
                eprintln!("  processing prompt token at pos {} -> id {}", pos, token);
            } else {
                eprintln!("  processing generated token at pos {} -> id {}", pos, token);
            }

            // Forward pass: process current token at current position
            // This adds the token to KV cache at position 'pos'
            if std::env::var("DEBUG").is_ok() {
                eprintln!("DEBUG: pos={}, token={}", pos, token);
            }

            // Check KV cache state before forward
            if std::env::var("DEBUG_KV").is_ok() && !self.kv_caches.is_empty() {
                eprintln!("  KV cache current_seq_len={}", self.kv_caches[0].current_seq_len);
            }

            let logits = self.forward(&[token], pos)?;

            // Check KV cache state after forward
            if std::env::var("DEBUG_KV").is_ok() && !self.kv_caches.is_empty() {
                eprintln!("  KV cache current_seq_len after={}", self.kv_caches[0].current_seq_len);
            }

            // Get logits for the last position
            let mut last_logits = logits[(logits.len() - self.config.vocab_size)..].to_vec();

            // Determine next token
            if pos < num_prompt_tokens - 1 {
                // Still processing prompt - skip sampling (already have next token in sequence)
                // Just continue to next position
            } else {
                // Past prompt - sample next token from logits
                // Debug: print top 5 logits before sampling
                if std::env::var("DEBUG_LOGITS").is_ok() {
                    let mut indexed: Vec<(usize, f32)> =
                        last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    eprintln!("  Top 5 logits:");
                    for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
                        eprintln!("    {}: token {} = {:.6}", i, idx, val);
                    }
                }

                // ALWAYS show top logits for debugging coherence
                let mut indexed: Vec<(usize, f32)> =
                    last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                eprintln!("🎯 Top 5 logits for sampling:");
                for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
                    eprintln!("    {}: token {} = {:.6}", i, idx, val);
                }

                let next = logits_processor
                    .sample(&mut last_logits)
                    .map_err(Error::ParseError)?;

                // ALWAYS show sampled token for debugging
                eprintln!("🎲 Sampled token: {} (logit: {:.6})", next, last_logits[next as usize]);

                // Check for EOS token
                if next == tokenizer.special_tokens().eos_token_id {
                    break;
                }

                // Add generated token to our sequence
                tokens.push(next);
            }

            // Advance position for next iteration
            pos += 1;
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

        // Create logits processor with configured sampling strategy
        let mut logits_processor = crate::sampling::LogitsProcessor::with_params(
            42, // seed - could be made configurable
            temperature as f64,
            top_p as f64,
            top_k,
            repetition_penalty,
        );

        // Clear KV cache for new generation
        self.clear_kv_cache();

        // Encode prompt
        let mut tokens = tokenizer.encode(prompt, false)?;

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
                next = logits_processor
                    .sample(&mut last_logits)
                    .map_err(Error::ParseError)?;

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

    pub fn rms_norm(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        let hidden_size = weight.len();
        let seq_len = input.len() / hidden_size;
        let mut output = vec![0.0; input.len()];

        for seq in 0..seq_len {
            let offset = seq * hidden_size;
            let slice = &input[offset..offset + hidden_size];

            // Compute RMS - FIXED: epsilon goes INSIDE sqrt like llama.cpp
            let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
            let mean = sum_sq / hidden_size as f32;
            let rms = (mean + self.config.rms_norm_eps).sqrt();

            // Normalize and scale
            for i in 0..hidden_size {
                output[offset + i] = (slice[i] / rms) * weight[i];
            }
        }

        Ok(output)
    }
}
