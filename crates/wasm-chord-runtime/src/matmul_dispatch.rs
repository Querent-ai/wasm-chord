//! Matmul dispatch for fused kernels
//!
//! This module provides intelligent dispatching to fused dequantization + matmul kernels
//! based on the weight format, enabling 2-4x speedups for quantized models.
//! Supports both CPU and GPU backends with automatic fallback.

use wasm_chord_core::error::Result;
use wasm_chord_cpu::{
    fused_dequant_matmul_q4k, fused_dequant_matmul_q5k, fused_dequant_matmul_q6k,
    fused_dequant_matmul_q8k, matmul_transposed,
};

#[cfg(feature = "webgpu")]
use wasm_chord_gpu::GpuBackend;

#[cfg(any(feature = "cuda", feature = "metal"))]
use wasm_chord_gpu::CandleGpuBackend;

use crate::weight_format::WeightFormat;

/// Dispatch matmul to appropriate kernel based on weight format and available backends
///
/// This function automatically selects the optimal implementation:
/// - **Quantized formats (Q4_K/Q5_K/Q6_K/Q8_K):** Fused dequant + matmul (2-4x faster)
/// - **F32:** Standard matmul
/// - **GPU backends:** When available, uses GPU acceleration
/// - **Automatic fallback:** CPU if GPU fails or unavailable
///
/// # Arguments
/// * `input` - Input activations [batch_size, k]
/// * `weights` - Weight matrix (any format) [n, k] (stored transposed)
/// * `batch_size` - Batch size (m)
/// * `k` - Input dimension
/// * `n` - Output dimension
/// * `gpu_backend` - Optional GPU backend (WebGPU/CUDA/Metal)
///
/// # Returns
/// Output activations [batch_size, n]
///
/// # Performance
/// - Q4_K CPU: ~8.6x faster than naive (measured)
/// - Q4_K GPU: ~10-20x faster (expected)
/// - Q5_K/Q6_K/Q8_K: ~2-4x faster
/// - F32: Baseline performance
pub fn dispatch_matmul(
    input: &[f32],
    weights: &WeightFormat,
    batch_size: usize,
    k: usize,
    n: usize,
    #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))] gpu_backend: Option<
        &dyn std::any::Any,
    >,
) -> Result<Vec<f32>> {
    // Debug: Log which path is taken
    if std::env::var("DEBUG_DISPATCH").is_ok() {
        eprintln!(
            "[dispatch_matmul] format={}, shape=[{}, {}, {}]",
            weights.format_name(),
            batch_size,
            k,
            n
        );
    }

    match weights {
        WeightFormat::F32(w) => {
            // Standard F32 matmul (weights stored transposed)
            let mut output = vec![0.0f32; batch_size * n];
            matmul_transposed(input, w, &mut output, batch_size, k, n)?;
            Ok(output)
        }
        WeightFormat::Q4K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                // Try WebGPU backend
                #[cfg(feature = "webgpu")]
                if let Some(webgpu) = gpu.downcast_ref::<GpuBackend>() {
                    if let Ok(result) =
                        webgpu.fused_dequant_matmul_q4k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }

                // Try CUDA/Metal backend
                #[cfg(any(feature = "cuda", feature = "metal"))]
                if let Some(candle_gpu) = gpu.downcast_ref::<CandleGpuBackend>() {
                    if let Ok(result) =
                        candle_gpu.fused_dequant_matmul_q4k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }
            }

            // CPU fallback: Fused Q4_K kernel (8.6x faster measured)
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q4k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q5K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                #[cfg(feature = "webgpu")]
                if let Some(webgpu) = gpu.downcast_ref::<GpuBackend>() {
                    if let Ok(result) =
                        webgpu.fused_dequant_matmul_q5k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }

                #[cfg(any(feature = "cuda", feature = "metal"))]
                if let Some(candle_gpu) = gpu.downcast_ref::<CandleGpuBackend>() {
                    if let Ok(result) =
                        candle_gpu.fused_dequant_matmul_q5k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }
            }

            // CPU fallback: Fused Q5_K kernel (2-3x faster)
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q5k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q6K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                #[cfg(feature = "webgpu")]
                if let Some(webgpu) = gpu.downcast_ref::<GpuBackend>() {
                    if let Ok(result) =
                        webgpu.fused_dequant_matmul_q6k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }

                #[cfg(any(feature = "cuda", feature = "metal"))]
                if let Some(candle_gpu) = gpu.downcast_ref::<CandleGpuBackend>() {
                    if let Ok(result) =
                        candle_gpu.fused_dequant_matmul_q6k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }
            }

            // CPU fallback: Fused Q6_K kernel (2-3x faster)
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q6k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
        WeightFormat::Q8K(blocks) => {
            // Try GPU first, fallback to CPU
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            if let Some(gpu) = gpu_backend {
                #[cfg(feature = "webgpu")]
                if let Some(webgpu) = gpu.downcast_ref::<GpuBackend>() {
                    if let Ok(result) =
                        webgpu.fused_dequant_matmul_q8k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }

                #[cfg(any(feature = "cuda", feature = "metal"))]
                if let Some(candle_gpu) = gpu.downcast_ref::<CandleGpuBackend>() {
                    if let Ok(result) =
                        candle_gpu.fused_dequant_matmul_q8k(blocks, input, batch_size, n, k)
                    {
                        return Ok(result);
                    }
                }
            }

            // CPU fallback: Fused Q8_K kernel (3-4x faster)
            let mut output = vec![0.0f32; batch_size * n];
            fused_dequant_matmul_q8k(blocks, input, &mut output, batch_size, n, k)?;
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_chord_core::quant::BlockQ4_K;

    #[test]
    fn test_dispatch_f32() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = WeightFormat::F32(vec![0.5; 8]); // [2, 4] transposed

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            4,
            2,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_dispatch_q4k() {
        let input = vec![0.5f32; 256];
        // Create 256 Q4_K blocks (one per output element)
        let block = BlockQ4_K {
            d: 0u16,           // f16 scale as u16
            dmin: 0u16,        // f16 min scale as u16
            scales: [0u8; 12], // Quantized scales
            qs: [0u8; 128],    // 4-bit quants (256/2 = 128 bytes)
        };
        let blocks = vec![block; 256]; // Need 256 blocks for 256x256 matmul
        let weights = WeightFormat::Q4K(blocks);

        let output = dispatch_matmul(
            &input,
            &weights,
            1,
            256,
            256,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        assert_eq!(output.len(), 256);
    }
}
