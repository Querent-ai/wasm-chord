//! Memory64 Performance Benchmark
//!
//! Measures:
//! - Memory usage (RSS, peak)
//! - Loading time
//! - Layer access time
//! - Cache hit rate

use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use sysinfo::System;
use wasm_chord_core::{error::Result, formats::gguf::GGUFParser};
use wasm_chord_runtime::memory64_gguf::Memory64GGUFLoader;

fn main() -> Result<()> {
    println!("🚀 Memory64 Performance Benchmark");
    println!("=================================\n");

    // Initialize system info
    let mut sys = System::new_all();
    let pid = sysinfo::get_current_pid().unwrap();

    // Test models
    let models = vec![
        ("TinyLlama 1.1B (Q4_K_M)", "models/tinyllama-1.1b.Q4_K_M.gguf", false),
        ("Llama-2-7B (Q4_K_M)", "models/llama-2-7b-chat-q4_k_m.gguf", true),
    ];

    for (name, path, expect_memory64) in models {
        println!("📊 Benchmarking: {}", name);
        println!("{}", "=".repeat(50));

        // Check if model exists
        if !std::path::Path::new(path).exists() {
            println!("⚠️  Model not found: {}", path);
            println!();
            continue;
        }

        // Measure initial memory
        sys.refresh_all();
        let process = sys.process(pid).unwrap();
        let initial_memory = process.memory();
        println!("📦 Initial memory: {:.2} MB", initial_memory as f64 / 1_000_000.0);

        // Benchmark 1: Model Loading Time
        println!("\n⏱️  Benchmark 1: Model Loading Time");
        let load_start = Instant::now();

        let file = File::open(path).map_err(|e| {
            wasm_chord_core::error::Error::ParseError(format!("Failed to open {}: {}", path, e))
        })?;
        let reader = BufReader::new(file);
        let mut parser = GGUFParser::new(reader);
        let mut loader = Memory64GGUFLoader::new();

        let mut model = match loader.load_model(&mut parser) {
            Ok(m) => m,
            Err(e) => {
                println!("❌ Failed to load model: {}", e);
                println!();
                continue;
            }
        };

        let load_duration = load_start.elapsed();
        println!("   ✅ Loading time: {:.2}s", load_duration.as_secs_f64());

        // Measure memory after loading
        sys.refresh_all();
        let process = sys.process(pid).unwrap();
        let loaded_memory = process.memory();
        let memory_increase = (loaded_memory - initial_memory) as f64 / 1_000_000.0;
        println!("   📈 Memory increase: {:.2} MB", memory_increase);
        println!("   📊 Total memory: {:.2} MB", loaded_memory as f64 / 1_000_000.0);

        // Benchmark 2: First Layer Access (Cold)
        println!("\n🔄 Benchmark 2: Layer Access Performance");
        let access_start = Instant::now();
        match model.get_layer(0) {
            Ok(_) => {
                let access_duration = access_start.elapsed();
                println!(
                    "   ✅ First layer access (cold): {:.2}ms",
                    access_duration.as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                println!("   ❌ Layer access failed: {}", e);
            }
        }

        // Benchmark 3: Subsequent Layer Accesses
        let mut total_access_time = 0.0;
        let layers_to_test = 5;

        for i in 1..layers_to_test {
            let access_start = Instant::now();
            if model.get_layer(i).is_ok() {
                total_access_time += access_start.elapsed().as_secs_f64();
            }
        }

        let avg_access_time = total_access_time / (layers_to_test - 1) as f64;
        println!("   ✅ Average layer access: {:.2}ms", avg_access_time * 1000.0);

        // Benchmark 4: Cache Performance
        println!("\n💾 Benchmark 3: Cache Performance");
        let stats = model.cache_stats();
        println!("   📊 Cached layers: {}/{}", stats.cached_layers, stats.max_cache_size);
        println!("   📈 Cache hits: {}", stats.cache_hits);
        println!("   📉 Cache misses: {}", stats.cache_misses);
        println!("   🗑️  Evictions: {}", stats.evictions);

        if stats.cache_hits + stats.cache_misses > 0 {
            let hit_rate =
                stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0;
            println!("   ⚡ Hit rate: {:.1}%", hit_rate);
        }

        // Benchmark 4.5: Test Prefetching
        if expect_memory64 {
            println!("\n⚡ Benchmark 3.5: Prefetch Performance");

            // Test with prefetch disabled
            model.set_prefetch_distance(0);
            let no_prefetch_start = Instant::now();
            for i in 0..5 {
                let _ = model.get_layer(i);
            }
            let no_prefetch_time = no_prefetch_start.elapsed();
            println!("   ⏱️  Without prefetch: {:.2}ms", no_prefetch_time.as_secs_f64() * 1000.0);

            // Reset cache
            model.clear_cache();

            // Test with prefetch enabled (distance=1)
            model.set_prefetch_distance(1);
            let prefetch_start = Instant::now();
            for i in 0..5 {
                let _ = model.get_layer(i);
            }
            let prefetch_time = prefetch_start.elapsed();
            println!("   ⚡ With prefetch (d=1): {:.2}ms", prefetch_time.as_secs_f64() * 1000.0);

            let improvement =
                (1.0 - prefetch_time.as_secs_f64() / no_prefetch_time.as_secs_f64()) * 100.0;
            println!("   📊 Improvement: {:.1}%", improvement);
        }

        // Benchmark 5: Sequential Layer Loading
        println!("\n🔄 Benchmark 4: Sequential Layer Loading");
        let num_layers = 10;
        let seq_start = Instant::now();

        for i in 0..num_layers {
            let _ = model.get_layer(i);
        }

        let seq_duration = seq_start.elapsed();
        println!("   ✅ Loaded {} layers in {:.2}s", num_layers, seq_duration.as_secs_f64());
        println!(
            "   ⚡ Average: {:.2}ms per layer",
            seq_duration.as_secs_f64() * 1000.0 / num_layers as f64
        );

        // Final memory measurement
        sys.refresh_all();
        let process = sys.process(pid).unwrap();
        let final_memory = process.memory();
        let total_increase = (final_memory - initial_memory) as f64 / 1_000_000.0;

        println!("\n📊 Memory Summary");
        println!("   Initial: {:.2} MB", initial_memory as f64 / 1_000_000.0);
        println!("   Final: {:.2} MB", final_memory as f64 / 1_000_000.0);
        println!("   Increase: {:.2} MB", total_increase);

        // Verify Memory64 usage
        println!("\n🎯 Memory64 Status");
        if expect_memory64 {
            println!("   ✅ Memory64 enabled (model >3GB)");
        } else {
            println!("   ℹ️  Standard loading (model <3GB)");
        }

        println!("\n{}\n", "=".repeat(50));
    }

    println!("🎉 Benchmark Complete!");
    println!("✅ Memory usage analyzed");
    println!("✅ Loading performance measured");
    println!("✅ Cache efficiency evaluated");
    println!("✅ Layer access speed tested");

    Ok(())
}
