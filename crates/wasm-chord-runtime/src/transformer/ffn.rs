//! Feed-forward network implementation

use wasm_chord_core::error::Result;
use wasm_chord_cpu::{matmul_f32, matmul_transposed};

#[cfg(feature = "webgpu")]
use wasm_chord_gpu::GpuBackend;

use super::TransformerConfig;
use crate::matmul_dispatch::dispatch_matmul;
use crate::weight_format::WeightFormat;

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
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        transposed_b: bool,
        #[cfg(feature = "webgpu")] _gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "webgpu")]
        if let Some(gpu) = _gpu {
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

    pub fn forward(
        &self,
        hidden_states: &[f32],
        weights: &FFNWeights,
        #[cfg(feature = "webgpu")] _gpu: Option<&GpuBackend>,
    ) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.config.hidden_size;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Gate projection using fused kernels
        let mut gate = dispatch_matmul(
            hidden_states,
            &weights.w_gate,
            seq_len,
            hidden_size,
            intermediate_size,
        )?;

        // Up projection using fused kernels
        let up =
            dispatch_matmul(hidden_states, &weights.w_up, seq_len, hidden_size, intermediate_size)?;

        // SwiGLU activation: silu(gate) * up
        // where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        for i in 0..gate.len() {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            let silu = gate[i] * sigmoid;
            gate[i] = silu * up[i];
        }

        // Down projection using fused kernels
        let output =
            dispatch_matmul(&gate, &weights.w_down, seq_len, intermediate_size, hidden_size)?;

        Ok(output)
    }
}

/// FFN weight matrices
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FFNWeights {
    /// Gate projection [hidden_size, intermediate_size]
    pub w_gate: WeightFormat,
    /// Up projection [hidden_size, intermediate_size]
    pub w_up: WeightFormat,
    /// Down projection [intermediate_size, hidden_size]
    pub w_down: WeightFormat,
}

#[allow(dead_code)]
impl FFNWeights {
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Self {
            w_gate: WeightFormat::new_f32(hidden_size * intermediate_size),
            w_up: WeightFormat::new_f32(hidden_size * intermediate_size),
            w_down: WeightFormat::new_f32(intermediate_size * hidden_size),
        }
    }
}
