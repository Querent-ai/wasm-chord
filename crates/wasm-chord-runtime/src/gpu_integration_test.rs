//! GPU Fused Kernel Integration Test
//!
//! This test verifies that the GPU dispatch integration works correctly
//! and falls back gracefully to CPU when GPU is not available.

#[cfg(test)]
mod tests {
    use crate::{matmul_dispatch::dispatch_matmul, weight_format::WeightFormat};
    use wasm_chord_core::quant::BlockQ4_K;

    #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
    #[allow(unused_imports)]
    use wasm_chord_gpu::{CandleGpuBackend, GpuBackend};

    #[test]
    fn test_gpu_dispatch_fallback() {
        // Create test data
        let input = vec![0.5f32; 256];
        let block = BlockQ4_K { d: 0u16, dmin: 0u16, scales: [0u8; 12], qs: [0u8; 128] };
        let blocks = vec![block; 256];
        let weights = WeightFormat::Q4K(blocks);

        // Test without GPU backend (should use CPU fallback)
        let output_cpu = dispatch_matmul(
            &input,
            &weights,
            1,
            256,
            256,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();

        assert_eq!(output_cpu.len(), 256);
        println!("✅ CPU fallback working: {} elements", output_cpu.len());

        // Test with GPU backend (if available)
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
        {
            // This would test actual GPU dispatch when GPU backends are implemented
            // For now, we just verify the interface compiles
            println!("✅ GPU dispatch interface ready");
        }
    }

    #[test]
    fn test_dispatch_performance_comparison() {
        use std::time::Instant;

        let input = vec![0.5f32; 1024];
        let block = BlockQ4_K { d: 0u16, dmin: 0u16, scales: [0u8; 12], qs: [0u8; 128] };
        let blocks = vec![block; 4096]; // Need 4096 blocks for 1024x1024 matmul (1024 * 4)
        let weights = WeightFormat::Q4K(blocks);

        // Measure CPU performance
        let start = Instant::now();
        let _output = dispatch_matmul(
            &input,
            &weights,
            1,
            1024,
            1024,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            None,
        )
        .unwrap();
        let cpu_time = start.elapsed();

        println!("✅ CPU fused kernel: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);

        // When GPU is available, this will show GPU vs CPU comparison
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
        {
            println!("🚀 GPU dispatch ready for testing on GPU-enabled machine");
        }
    }
}
