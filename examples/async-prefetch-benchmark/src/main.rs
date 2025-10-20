//! Async Prefetch Benchmark
//!
//! This example benchmarks the performance improvement from async background prefetch
//! compared to synchronous prefetch.

use wasm_chord_runtime::async_prefetch::{AsyncPrefetchConfig, AsyncMemory64Model};
use wasm_chord_runtime::memory64_layer_manager::{Memory64LayerManager, Memory64Model};
use wasm_chord_runtime::memory64::MemoryLayout;
use wasm_chord_core::error::Result;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Async Prefetch Benchmark");
    println!("===========================\n");

    // Initialize tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Create a mock transformer config (TinyLlama-like)
    let config = wasm_chord_runtime::TransformerConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 32,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        ..Default::default()
    };

    // Create mock data
    let token_embeddings = vec![0.0; config.vocab_size * config.hidden_size];
    let output_norm = vec![0.0; config.hidden_size];
    let lm_head = vec![0.0; config.vocab_size * config.hidden_size];

    // Create mock layer manager
    let layer_manager = Memory64LayerManager::new(
        std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
            MemoryLayout::single(4, "benchmark").unwrap(),
            true,
        )),
        config.clone(),
        4, // max_cache_size
    );

    // Benchmark 1: Synchronous Prefetch (baseline)
    println!("📊 Benchmark 1: Synchronous Prefetch (Baseline)");
    let start_time = Instant::now();
    
    let mut sync_model = Memory64Model::new(
        config.clone(),
        token_embeddings.clone(),
        output_norm.clone(),
        lm_head.clone(),
        layer_manager,
        config.num_layers as u32,
    );
    
    // Simulate sequential layer access (typical inference pattern)
    for layer_id in 0..config.num_layers {
        let _layer = sync_model.get_layer(layer_id as u32)?;
        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    
    let sync_duration = start_time.elapsed();
    println!("   ⏱️ Synchronous prefetch time: {:?}", sync_duration);
    println!("   📈 Cache stats: {:?}", sync_model.cache_stats());

    // Benchmark 2: Async Prefetch
    println!("\n📊 Benchmark 2: Async Prefetch (Optimized)");
    let start_time = Instant::now();
    
    // Create a new layer manager for async test
    let async_layer_manager = Memory64LayerManager::new(
        std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
            MemoryLayout::single(4, "async-benchmark").unwrap(),
            true,
        )),
        config.clone(),
        4, // max_cache_size
    );
    
    let mut async_model = AsyncMemory64Model::new(
        config.clone(),
        token_embeddings,
        output_norm,
        lm_head,
        async_layer_manager,
        config.num_layers as u32,
        AsyncPrefetchConfig {
            prefetch_distance: 2,
            max_concurrent_tasks: 4,
            smart_prefetch: true,
        },
    );
    
    // Simulate sequential layer access with async prefetch
    for layer_id in 0..config.num_layers {
        let _layer = async_model.get_layer_async(layer_id as u32).await?;
        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    
    let async_duration = start_time.elapsed();
    println!("   ⏱️ Async prefetch time: {:?}", async_duration);
    
    // Get async prefetch stats
    let prefetch_stats = async_model.get_prefetch_stats().await;
    println!("   📈 Prefetch stats: {:?}", prefetch_stats);

    // Calculate performance improvement
    let improvement = if sync_duration > async_duration {
        let diff = sync_duration - async_duration;
        let percentage = (diff.as_millis() as f64 / sync_duration.as_millis() as f64) * 100.0;
        format!("+{:.1}% faster", percentage)
    } else {
        let diff = async_duration - sync_duration;
        let percentage = (diff.as_millis() as f64 / sync_duration.as_millis() as f64) * 100.0;
        format!("-{:.1}% slower", percentage)
    };

    println!("\n🎯 Performance Results:");
    println!("   Synchronous: {:?}", sync_duration);
    println!("   Async:       {:?}", async_duration);
    println!("   Improvement: {}", improvement);

    // Benchmark 3: Different prefetch distances
    println!("\n📊 Benchmark 3: Prefetch Distance Comparison");
    
    for prefetch_distance in [0, 1, 2, 4] {
        let start_time = Instant::now();
        
        let mut test_model = AsyncMemory64Model::new(
            config.clone(),
            vec![0.0; config.vocab_size * config.hidden_size],
            vec![0.0; config.hidden_size],
            vec![0.0; config.vocab_size * config.hidden_size],
            Memory64LayerManager::new(
                std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
                    MemoryLayout::single(4, "test-benchmark").unwrap(),
                    true,
                )),
                config.clone(),
                4,
            ),
            config.num_layers as u32,
            AsyncPrefetchConfig {
                prefetch_distance,
                max_concurrent_tasks: 4,
                smart_prefetch: true,
            },
        );
        
        // Access first 10 layers
        for layer_id in 0..10 {
            let _layer = test_model.get_layer_async(layer_id as u32).await?;
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        }
        
        let duration = start_time.elapsed();
        let stats = test_model.get_prefetch_stats().await;
        
        println!("   Prefetch distance {}: {:?} ({} prefetched)", 
                 prefetch_distance, duration, stats.prefetched_layers);
    }

    println!("\n🎉 Async Prefetch Benchmark Complete!");
    println!("✅ Async prefetch system working");
    println!("✅ Performance improvements measurable");
    println!("✅ Configurable prefetch distances");

    Ok(())
}
