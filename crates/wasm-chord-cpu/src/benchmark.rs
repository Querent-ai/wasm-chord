//! Simple benchmark for CPU optimizations
//!
//! This benchmark tests the memory pool optimized fused kernel
//! against the baseline implementation.

use crate::fused::{fused_dequant_matmul_q4k, fused_dequant_matmul_q4k_pooled};
use std::time::Instant;
use wasm_chord_core::quant::{BlockQ4_K, QK_K};

/// Run a simple benchmark comparing baseline vs optimized
pub fn run_simple_benchmark() {
    println!("🚀 Running simple CPU optimization benchmark...");

    // Test configuration
    let batch_size = 8;
    let num_output_features = 2048;
    let k = 2048;
    let num_iterations = 50;
    let warmup_iterations = 5;

    println!(
        "📊 Config: batch_size={}, output_features={}, k={}",
        batch_size, num_output_features, k
    );

    // Generate test data
    let (quantized_weights, input, mut output_baseline, mut output_optimized) =
        generate_test_data(batch_size, num_output_features, k);

    println!("📦 Generated {} Q4_K blocks, {} input values", quantized_weights.len(), input.len());

    // Warmup runs
    println!("🔥 Warming up...");
    for _ in 0..warmup_iterations {
        let _ = fused_dequant_matmul_q4k(
            &quantized_weights,
            &input,
            &mut output_baseline,
            batch_size,
            num_output_features,
            k,
        );

        let _ = fused_dequant_matmul_q4k_pooled(
            &quantized_weights,
            &input,
            &mut output_optimized,
            batch_size,
            num_output_features,
            k,
        );
    }

    // Benchmark baseline
    println!("📈 Benchmarking baseline implementation...");
    let baseline_start = Instant::now();
    for _ in 0..num_iterations {
        let _ = fused_dequant_matmul_q4k(
            &quantized_weights,
            &input,
            &mut output_baseline,
            batch_size,
            num_output_features,
            k,
        );
    }
    let baseline_duration = baseline_start.elapsed();
    let baseline_time_ms = baseline_duration.as_secs_f64() * 1000.0 / num_iterations as f64;

    // Benchmark optimized version
    println!("⚡ Benchmarking optimized implementation...");
    let optimized_start = Instant::now();
    for _ in 0..num_iterations {
        let _ = fused_dequant_matmul_q4k_pooled(
            &quantized_weights,
            &input,
            &mut output_optimized,
            batch_size,
            num_output_features,
            k,
        );
    }
    let optimized_duration = optimized_start.elapsed();
    let optimized_time_ms = optimized_duration.as_secs_f64() * 1000.0 / num_iterations as f64;

    // Calculate speedup
    let speedup = baseline_time_ms / optimized_time_ms;

    // Verify correctness
    verify_correctness(&output_baseline, &output_optimized);

    // Print results
    println!("\n🎉 BENCHMARK RESULTS");
    println!("===================");
    println!("📊 Baseline time:    {:.2} ms", baseline_time_ms);
    println!("⚡ Optimized time:    {:.2} ms", optimized_time_ms);
    println!("🚀 Speedup:          {:.2}x", speedup);

    println!("\n🎯 PERFORMANCE ANALYSIS");
    println!("========================");

    if speedup >= 1.2 {
        println!("🏆 EXCELLENT: Significant speedup achieved!");
    } else if speedup >= 1.1 {
        println!("🎯 GOOD: Modest speedup achieved");
    } else if speedup >= 1.0 {
        println!("📈 IMPROVEMENT: At least no regression");
    } else {
        println!("⚠️  NEEDS WORK: Performance regression detected");
    }
}

/// Generate test data for benchmarking
fn generate_test_data(
    batch_size: usize,
    num_output_features: usize,
    k: usize,
) -> (Vec<BlockQ4_K>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_blocks_per_row = k / QK_K;
    let total_blocks = num_output_features * num_blocks_per_row;

    // Generate quantized weights
    let mut quantized_weights = Vec::with_capacity(total_blocks);
    for i in 0..total_blocks {
        let mut block = BlockQ4_K { d: 0u16, dmin: 0u16, scales: [0u8; 12], qs: [0u8; 128] };

        // Fill with test data
        for j in 0..128 {
            block.qs[j] = ((i + j) as u8) % 16;
        }
        for j in 0..12 {
            block.scales[j] = ((i + j) as u8) % 8;
        }

        quantized_weights.push(block);
    }

    // Generate input data
    let input_size = batch_size * k;
    let input: Vec<f32> = (0..input_size).map(|i| (i as f32) * 0.01).collect();

    // Initialize output buffers
    let output_size = batch_size * num_output_features;
    let output_baseline = vec![0.0f32; output_size];
    let output_optimized = vec![0.0f32; output_size];

    (quantized_weights, input, output_baseline, output_optimized)
}

/// Verify correctness of optimized implementation
fn verify_correctness(baseline: &[f32], optimized: &[f32]) {
    let tolerance = 1e-4;
    let mut max_error: f32 = 0.0;

    for (i, (baseline_val, optimized_val)) in baseline.iter().zip(optimized.iter()).enumerate() {
        let error = (baseline_val - optimized_val).abs();
        max_error = max_error.max(error);

        if error > tolerance {
            panic!(
                "Correctness check failed at index {}: baseline={}, optimized={}, error={}",
                i, baseline_val, optimized_val, error
            );
        }
    }

    println!("✅ Correctness verified: max_error={:.6}", max_error);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_data() {
        let (weights, input, output1, output2) = generate_test_data(2, 4, 256);
        assert_eq!(weights.len(), 4); // 4 output features * 1 block per row
        assert_eq!(input.len(), 512); // 2 batch * 256 k
        assert_eq!(output1.len(), 8); // 2 batch * 4 output features
        assert_eq!(output2.len(), 8);
    }

    #[test]
    fn test_run_benchmark() {
        run_simple_benchmark();
    }
}
