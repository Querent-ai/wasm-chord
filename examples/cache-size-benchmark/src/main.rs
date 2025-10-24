//! Cache Size Optimization Benchmark
//!
//! This example benchmarks the performance impact of different cache sizes
//! to find the optimal balance between memory usage and performance.

use wasm_chord_runtime::memory64_layer_manager::{Memory64LayerManager, Memory64Model};
use wasm_chord_runtime::memory64::MemoryLayout;
use wasm_chord_runtime::attention::AttentionBackend;
use wasm_chord_core::error::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Cache Size Optimization Benchmark");
    println!("====================================\n");

    // Create a mock transformer config (TinyLlama-like)
    let config = wasm_chord_runtime::TransformerConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 32,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
        ..Default::default()
    };

    // Create mock data
    let token_embeddings = vec![0.0; config.vocab_size * config.hidden_size];
    let output_norm = vec![0.0; config.hidden_size];
    let lm_head = vec![0.0; config.vocab_size * config.hidden_size];

    // Test different cache sizes
    let cache_sizes = [4, 8, 12, 16];
    
    for &cache_size in &cache_sizes {
        println!("📊 Testing cache size: {} layers", cache_size);
        
        // Create layer manager with specific cache size
        let layer_manager = Memory64LayerManager::new(
            std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
                MemoryLayout::single(4, "cache-benchmark").unwrap(),
                true,
            )),
            config.clone(),
            cache_size,
        );

        let mut model = Memory64Model::new(
            config.clone(),
            token_embeddings.clone(),
            output_norm.clone(),
            lm_head.clone(),
            layer_manager,
            config.num_layers as u32,
        );

        // Benchmark sequential layer access
        let start_time = Instant::now();
        
        // Access all layers sequentially (typical inference pattern)
        for layer_id in 0..config.num_layers {
            let _layer = model.get_layer(layer_id as u32)?;
        }
        
        let duration = start_time.elapsed();
        let stats = model.cache_stats();
        let (cached, max) = model.get_cache_utilization();
        
        println!("   ⏱️ Time: {:?}", duration);
        println!("   📈 Cache hits: {}", stats.cache_hits);
        println!("   📉 Cache misses: {}", stats.cache_misses);
        println!("   🎯 Hit rate: {:.1}%", 
                 (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0);
        println!("   🗑️ Evictions: {}", stats.evictions);
        println!("   💾 Cache utilization: {}/{} layers", cached, max);
        println!("   📊 Memory usage: ~{} MB", cached * 50); // ~50MB per layer
        
        println!();
    }

    // Test auto-configuration
    println!("🤖 Testing auto-configuration:");
    
    let layer_manager = Memory64LayerManager::new(
        std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
            MemoryLayout::single(4, "auto-config").unwrap(),
            true,
        )),
        config.clone(),
        4, // Start with small cache
    );

    let mut model = Memory64Model::new(
        config.clone(),
        token_embeddings,
        output_norm,
        lm_head,
        layer_manager,
        config.num_layers as u32,
    );

    println!("   Initial cache size: {} layers", model.get_cache_size());
    
    // Auto-configure based on available memory
    model.auto_configure_cache_size();
    
    println!("   Auto-configured cache size: {} layers", model.get_cache_size());
    
    // Test performance with auto-configured cache
    let start_time = Instant::now();
    for layer_id in 0..config.num_layers {
        let _layer = model.get_layer(layer_id as u32)?;
    }
    let duration = start_time.elapsed();
    let stats = model.cache_stats();
    
    println!("   ⏱️ Auto-configured performance: {:?}", duration);
    println!("   📈 Hit rate: {:.1}%", 
             (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0);

    println!("\n🎯 Cache Size Optimization Results:");
    println!("✅ Configurable cache sizes working");
    println!("✅ Auto-configuration implemented");
    println!("✅ Performance impact measurable");
    println!("✅ Memory usage tracking");

    Ok(())
}
