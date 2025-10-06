//! Performance Benchmark for wasm-chord
//!
//! Benchmarks CPU and GPU matmul performance to demonstrate speedup.
//!
//! Usage:
//!   # CPU only
//!   cargo run --release --manifest-path examples/benchmark/Cargo.toml
//!
//!   # With GPU
//!   cargo run --release --features gpu --manifest-path examples/benchmark/Cargo.toml

use std::time::Instant;
use wasm_chord_core::error::Result;

// Benchmark configurations
const SMALL_MATRIX: (usize, usize, usize) = (128, 128, 128);
const MEDIUM_MATRIX: (usize, usize, usize) = (512, 512, 512);
const LARGE_MATRIX: (usize, usize, usize) = (1024, 1024, 1024);

fn benchmark_cpu_matmul(m: usize, k: usize, n: usize) -> Result<f64> {
    use wasm_chord_cpu::matmul_f32;

    // Create test matrices
    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // Warmup
    matmul_f32(&a, &b, &mut c, m, k, n)?;

    // Benchmark
    let iterations = 3;
    let start = Instant::now();

    for _ in 0..iterations {
        matmul_f32(&a, &b, &mut c, m, k, n)?;
    }

    let duration = start.elapsed();
    let avg_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;

    Ok(avg_ms)
}

#[cfg(feature = "gpu")]
fn benchmark_gpu_matmul(m: usize, k: usize, n: usize) -> Result<f64> {
    use wasm_chord_gpu::GpuBackend;

    // Initialize GPU
    let gpu = pollster::block_on(GpuBackend::new())?;

    // Create test matrices
    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];

    // Warmup
    let _ = gpu.matmul(&a, &b, m as u32, k as u32, n as u32)?;

    // Benchmark
    let iterations = 3;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = gpu.matmul(&a, &b, m as u32, k as u32, n as u32)?;
    }

    let duration = start.elapsed();
    let avg_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;

    Ok(avg_ms)
}

fn calculate_gflops(m: usize, k: usize, n: usize, time_ms: f64) -> f64 {
    let operations = 2.0 * m as f64 * k as f64 * n as f64; // 2 * M * K * N for matmul
    let gflops = (operations / 1e9) / (time_ms / 1000.0);
    gflops
}

fn main() -> Result<()> {
    println!("🚀 wasm-chord Performance Benchmark");
    println!("====================================\n");

    let test_cases = vec![
        ("Small (128x128x128)", SMALL_MATRIX),
        ("Medium (512x512x512)", MEDIUM_MATRIX),
        ("Large (1024x1024x1024)", LARGE_MATRIX),
    ];

    println!("Running CPU benchmarks...\n");

    for (name, (m, k, n)) in &test_cases {
        print!("{}: ", name);
        match benchmark_cpu_matmul(*m, *k, *n) {
            Ok(time_ms) => {
                let gflops = calculate_gflops(*m, *k, *n, time_ms);
                println!("{:.2} ms ({:.2} GFLOPS)", time_ms, gflops);
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    #[cfg(feature = "gpu")]
    {
        println!("\n Running GPU benchmarks...\n");

        for (name, (m, k, n)) in &test_cases {
            print!("{}: ", name);
            match benchmark_gpu_matmul(*m, *k, *n) {
                Ok(time_ms) => {
                    let gflops = calculate_gflops(*m, *k, *n, time_ms);
                    println!("{:.2} ms ({:.2} GFLOPS)", time_ms, gflops);
                }
                Err(e) => println!("Error: {}", e),
            }
        }

        println!("\n📊 Speedup Analysis\n");

        for (name, (m, k, n)) in &test_cases {
            if let (Ok(cpu_time), Ok(gpu_time)) =
                (benchmark_cpu_matmul(*m, *k, *n), benchmark_gpu_matmul(*m, *k, *n))
            {
                let speedup = cpu_time / gpu_time;
                println!("{}: {:.2}x speedup (GPU)", name, speedup);
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n💡 Tip: Run with --features gpu to benchmark GPU performance:");
        println!("   cargo run --release --features gpu --manifest-path examples/benchmark/Cargo.toml");
    }

    println!("\n✅ Benchmark complete!");

    Ok(())
}
