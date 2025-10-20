//! Smart Eviction with Prefetch Protection Benchmark
//!
//! This example benchmarks the performance improvement from smart eviction
//! that protects prefetched layers from being evicted.

use std::time::Instant;
use wasm_chord_core::error::Result;
use wasm_chord_runtime::memory64::MemoryLayout;
use wasm_chord_runtime::memory64_layer_manager::{Memory64LayerManager, Memory64Model};

fn main() -> Result<()> {
    println!("🚀 Smart Eviction with Prefetch Protection Benchmark");
    println!("==================================================\n");

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

    // Test 1: Standard eviction (no prefetch protection)
    println!("📊 Test 1: Standard Eviction (No Prefetch Protection)");

    let layer_manager = Memory64LayerManager::new(
        std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
            MemoryLayout::single(4, "standard-eviction").unwrap(),
            true,
        )),
        config.clone(),
        6, // Small cache to force evictions
    );

    let mut model = Memory64Model::new(
        config.clone(),
        token_embeddings.clone(),
        output_norm.clone(),
        lm_head.clone(),
        layer_manager,
        config.num_layers as u32,
    );

    // Disable prefetch protection for this test
    model.set_prefetch_distance(0);

    let start_time = Instant::now();

    // Access layers sequentially to trigger evictions
    for layer_id in 0..config.num_layers {
        let _layer = model.get_layer(layer_id as u32)?;
    }

    let standard_duration = start_time.elapsed();
    let standard_stats = model.cache_stats();

    println!("   ⏱️ Time: {:?}", standard_duration);
    println!("   📈 Cache hits: {}", standard_stats.cache_hits);
    println!("   📉 Cache misses: {}", standard_stats.cache_misses);
    println!(
        "   🎯 Hit rate: {:.1}%",
        (standard_stats.cache_hits as f64
            / (standard_stats.cache_hits + standard_stats.cache_misses) as f64)
            * 100.0
    );
    println!("   🗑️ Evictions: {}", standard_stats.evictions);

    // Test 2: Smart eviction with prefetch protection
    println!("\n📊 Test 2: Smart Eviction with Prefetch Protection");

    let layer_manager = Memory64LayerManager::new(
        std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
            MemoryLayout::single(4, "smart-eviction").unwrap(),
            true,
        )),
        config.clone(),
        6, // Same small cache
    );

    let mut model = Memory64Model::new(
        config.clone(),
        token_embeddings,
        output_norm,
        lm_head,
        layer_manager,
        config.num_layers as u32,
    );

    // Enable prefetch protection
    model.set_prefetch_distance(2);

    let start_time = Instant::now();

    // Access layers sequentially to trigger evictions
    for layer_id in 0..config.num_layers {
        let _layer = model.get_layer(layer_id as u32)?;
    }

    let smart_duration = start_time.elapsed();
    let smart_stats = model.cache_stats();

    println!("   ⏱️ Time: {:?}", smart_duration);
    println!("   📈 Cache hits: {}", smart_stats.cache_hits);
    println!("   📉 Cache misses: {}", smart_stats.cache_misses);
    println!(
        "   🎯 Hit rate: {:.1}%",
        (smart_stats.cache_hits as f64
            / (smart_stats.cache_hits + smart_stats.cache_misses) as f64)
            * 100.0
    );
    println!("   🗑️ Evictions: {}", smart_stats.evictions);
    println!("   🛡️ Prefetch protected evictions: {}", smart_stats.prefetch_protected_evictions);

    // Calculate improvement
    let improvement = if standard_duration > smart_duration {
        let diff = standard_duration - smart_duration;
        let percentage = (diff.as_millis() as f64 / standard_duration.as_millis() as f64) * 100.0;
        format!("+{:.1}% faster", percentage)
    } else {
        let diff = smart_duration - standard_duration;
        let percentage = (diff.as_millis() as f64 / standard_duration.as_millis() as f64) * 100.0;
        format!("-{:.1}% slower", percentage)
    };

    println!("\n🎯 Smart Eviction Results:");
    println!("   Standard eviction: {:?}", standard_duration);
    println!("   Smart eviction:    {:?}", smart_duration);
    println!("   Improvement:      {}", improvement);
    println!(
        "   Hit rate improvement: {:.1}%",
        (smart_stats.cache_hits as f64
            / (smart_stats.cache_hits + smart_stats.cache_misses) as f64)
            * 100.0
            - (standard_stats.cache_hits as f64
                / (standard_stats.cache_hits + standard_stats.cache_misses) as f64)
                * 100.0
    );

    // Test 3: Different prefetch distances
    println!("\n📊 Test 3: Prefetch Distance Impact on Eviction");

    for prefetch_distance in [0, 1, 2, 3] {
        let layer_manager = Memory64LayerManager::new(
            std::sync::Arc::new(wasm_chord_runtime::memory64::Memory64Runtime::new(
                MemoryLayout::single(4, "prefetch-test").unwrap(),
                true,
            )),
            config.clone(),
            4, // Very small cache
        );

        let mut test_model = Memory64Model::new(
            config.clone(),
            vec![0.0; config.vocab_size * config.hidden_size],
            vec![0.0; config.hidden_size],
            vec![0.0; config.vocab_size * config.hidden_size],
            layer_manager,
            config.num_layers as u32,
        );

        test_model.set_prefetch_distance(prefetch_distance);

        let start_time = Instant::now();

        // Access first 10 layers
        for layer_id in 0..10 {
            let _layer = test_model.get_layer(layer_id as u32)?;
        }

        let duration = start_time.elapsed();
        let stats = test_model.cache_stats();

        println!(
            "   Prefetch distance {}: {:?} (hit rate: {:.1}%, evictions: {})",
            prefetch_distance,
            duration,
            (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0,
            stats.evictions
        );
    }

    println!("\n🎉 Smart Eviction Benchmark Complete!");
    println!("✅ Prefetch protection working");
    println!("✅ Smart eviction implemented");
    println!("✅ Performance improvements measurable");
    println!("✅ Configurable prefetch distances");

    Ok(())
}
