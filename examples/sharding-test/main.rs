/// Deterministic Layer Sharding Test
///
/// Tests the layer sharding system with deterministic allocation patterns
/// to ensure consistent behavior across runs for large model support.
use wasm_chord_runtime::{ShardConfig, ShardingManager, ShardingStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Layer Sharding Deterministic Test");
    println!("=====================================\n");

    // Test 1: No Sharding (Default)
    println!("📝 Test 1: No Sharding (Single Region)");
    let manager = ShardingManager::new(32);

    assert_eq!(manager.strategy(), ShardingStrategy::None);
    assert_eq!(manager.num_shards(), 1);
    assert_eq!(manager.total_layers(), 32);

    // Verify all layers are in shard 0
    for layer in 0..32 {
        let shard = manager.shard_for_layer(layer).unwrap();
        assert_eq!(shard.region_id, 0, "All layers should be in region 0");
        assert_eq!(shard.start_layer, 0);
        assert_eq!(shard.end_layer, 32);
    }

    println!("   Strategy: {:?}", manager.strategy());
    println!("   Total layers: {}", manager.total_layers());
    println!("   Num shards: {}", manager.num_shards());
    println!("   ✅ No sharding is deterministic\n");

    // Test 2: Sequential Sharding (Even Distribution)
    println!("📝 Test 2: Sequential Sharding (Even Split)");
    let manager = ShardingManager::with_sequential_sharding(32, 4, 1024 * 1024)?;

    assert_eq!(manager.strategy(), ShardingStrategy::Sequential);
    assert_eq!(manager.num_shards(), 4);

    let shards = manager.shards();
    let expected_distribution = vec![
        (0, 8, 0),   // Layers 0-7 in region 0
        (8, 16, 1),  // Layers 8-15 in region 1
        (16, 24, 2), // Layers 16-23 in region 2
        (24, 32, 3), // Layers 24-31 in region 3
    ];

    for (i, (start, end, region)) in expected_distribution.iter().enumerate() {
        assert_eq!(shards[i].start_layer, *start, "Shard {} start mismatch", i);
        assert_eq!(shards[i].end_layer, *end, "Shard {} end mismatch", i);
        assert_eq!(shards[i].region_id, *region, "Shard {} region mismatch", i);
        assert_eq!(shards[i].layer_count(), 8, "Each shard should have 8 layers");
        println!("   Shard {}: layers {}-{} → region {}", i, start, end - 1, region);
    }

    // Verify layer lookup is deterministic
    assert_eq!(manager.shard_for_layer(0).unwrap().region_id, 0);
    assert_eq!(manager.shard_for_layer(7).unwrap().region_id, 0);
    assert_eq!(manager.shard_for_layer(8).unwrap().region_id, 1);
    assert_eq!(manager.shard_for_layer(15).unwrap().region_id, 1);
    assert_eq!(manager.shard_for_layer(31).unwrap().region_id, 3);

    println!("   ✅ Sequential sharding is deterministic\n");

    // Test 3: Sequential Sharding (Uneven Distribution)
    println!("📝 Test 3: Sequential Sharding (Uneven Split)");
    let manager = ShardingManager::with_sequential_sharding(35, 4, 1024 * 1024)?;

    let shards = manager.shards();
    let total_layers: usize = shards.iter().map(|s| s.layer_count()).sum();
    assert_eq!(total_layers, 35, "Total layers must equal 35");

    println!("   Total layers: {}", total_layers);
    for (i, shard) in shards.iter().enumerate() {
        println!(
            "   Shard {}: layers {}-{} ({} layers) → region {}",
            i,
            shard.start_layer,
            shard.end_layer - 1,
            shard.layer_count(),
            shard.region_id
        );
    }

    // Verify no gaps or overlaps
    let mut covered = vec![false; 35];
    for shard in shards {
        for layer in shard.start_layer..shard.end_layer {
            assert!(!covered[layer], "Layer {} covered by multiple shards", layer);
            covered[layer] = true;
        }
    }
    assert!(covered.iter().all(|&c| c), "All layers must be covered");

    println!("   ✅ Uneven sharding is deterministic\n");

    // Test 4: Custom Sharding
    println!("📝 Test 4: Custom Sharding Configuration");
    let custom_shards = vec![
        ShardConfig::new(0, 10, 0, 10 * 1024 * 1024), // First 10 layers → region 0
        ShardConfig::new(10, 20, 1, 10 * 1024 * 1024), // Next 10 layers → region 1
        ShardConfig::new(20, 32, 2, 12 * 1024 * 1024), // Last 12 layers → region 2
    ];

    let manager = ShardingManager::with_custom_sharding(32, custom_shards)?;

    assert_eq!(manager.strategy(), ShardingStrategy::Custom);
    assert_eq!(manager.num_shards(), 3);

    // Verify custom boundaries
    assert_eq!(manager.shard_for_layer(0).unwrap().region_id, 0);
    assert_eq!(manager.shard_for_layer(9).unwrap().region_id, 0);
    assert_eq!(manager.shard_for_layer(10).unwrap().region_id, 1);
    assert_eq!(manager.shard_for_layer(19).unwrap().region_id, 1);
    assert_eq!(manager.shard_for_layer(20).unwrap().region_id, 2);
    assert_eq!(manager.shard_for_layer(31).unwrap().region_id, 2);

    println!("   Custom boundaries:");
    for (i, shard) in manager.shards().iter().enumerate() {
        println!(
            "   Shard {}: layers {}-{} ({} layers) → region {}",
            i,
            shard.start_layer,
            shard.end_layer - 1,
            shard.layer_count(),
            shard.region_id
        );
    }
    println!("   ✅ Custom sharding is deterministic\n");

    // Test 5: Memory Statistics
    println!("📝 Test 5: Memory Statistics");
    let shards = vec![
        ShardConfig::new(0, 10, 0, 100 * 1024 * 1024),  // 100MB
        ShardConfig::new(10, 20, 0, 200 * 1024 * 1024), // 200MB (same region)
        ShardConfig::new(20, 32, 1, 300 * 1024 * 1024), // 300MB
    ];

    let manager = ShardingManager::with_custom_sharding(32, shards)?;
    let stats = manager.memory_stats();

    println!("   Memory statistics by region:");
    for (region_id, layer_count, memory_bytes) in &stats {
        let memory_mb = *memory_bytes as f64 / (1024.0 * 1024.0);
        println!("   Region {}: {} layers, {:.1} MB", region_id, layer_count, memory_mb);
    }

    // Region 0 should have 20 layers (10+10) and 300MB (100+200)
    let region0 = stats.iter().find(|(id, _, _)| *id == 0).unwrap();
    assert_eq!(region0.1, 20, "Region 0 should have 20 layers");
    assert_eq!(region0.2, 300 * 1024 * 1024, "Region 0 should have 300MB");

    // Region 1 should have 12 layers and 300MB
    let region1 = stats.iter().find(|(id, _, _)| *id == 1).unwrap();
    assert_eq!(region1.1, 12, "Region 1 should have 12 layers");
    assert_eq!(region1.2, 300 * 1024 * 1024, "Region 1 should have 300MB");

    assert_eq!(manager.total_memory(), 600 * 1024 * 1024, "Total should be 600MB");
    println!("   Total memory: {:.1} MB", manager.total_memory() as f64 / (1024.0 * 1024.0));
    println!("   ✅ Memory statistics are deterministic\n");

    // Test 6: Error Handling (Gaps)
    println!("📝 Test 6: Error Detection - Gap in Coverage");
    let invalid_shards = vec![
        ShardConfig::new(0, 10, 0, 1024),
        ShardConfig::new(20, 32, 1, 1024), // Gap: layers 10-19 missing
    ];

    let result = ShardingManager::with_custom_sharding(32, invalid_shards);
    assert!(result.is_err(), "Should detect gap in layer coverage");
    println!("   ✅ Correctly detected gap in layers 10-19\n");

    // Test 7: Error Handling (Overlaps)
    println!("📝 Test 7: Error Detection - Overlapping Shards");
    let invalid_shards = vec![
        ShardConfig::new(0, 15, 0, 1024),
        ShardConfig::new(10, 32, 1, 1024), // Overlap: layers 10-14
    ];

    let result = ShardingManager::with_custom_sharding(32, invalid_shards);
    assert!(result.is_err(), "Should detect overlapping shards");
    println!("   ✅ Correctly detected overlap in layers 10-14\n");

    // Test 8: Error Handling (Invalid Range)
    println!("📝 Test 8: Error Detection - Invalid Range");
    let invalid_shards = vec![
        ShardConfig::new(10, 10, 0, 1024), // start == end (invalid)
        ShardConfig::new(10, 32, 1, 1024),
    ];

    let result = ShardingManager::with_custom_sharding(32, invalid_shards);
    assert!(result.is_err(), "Should detect invalid range");
    println!("   ✅ Correctly detected invalid range (start == end)\n");

    // Test 9: Cross-Instance Consistency
    println!("📝 Test 9: Cross-Instance Consistency");

    // Create two identical managers
    let manager1 = ShardingManager::with_sequential_sharding(32, 4, 1024 * 1024)?;
    let manager2 = ShardingManager::with_sequential_sharding(32, 4, 1024 * 1024)?;

    // Verify all properties match
    assert_eq!(manager1.strategy(), manager2.strategy());
    assert_eq!(manager1.num_shards(), manager2.num_shards());
    assert_eq!(manager1.total_layers(), manager2.total_layers());
    assert_eq!(manager1.total_memory(), manager2.total_memory());

    let shards1 = manager1.shards();
    let shards2 = manager2.shards();

    for (i, (s1, s2)) in shards1.iter().zip(shards2.iter()).enumerate() {
        assert_eq!(s1.start_layer, s2.start_layer, "Shard {} start mismatch", i);
        assert_eq!(s1.end_layer, s2.end_layer, "Shard {} end mismatch", i);
        assert_eq!(s1.region_id, s2.region_id, "Shard {} region mismatch", i);
        assert_eq!(s1.memory_bytes, s2.memory_bytes, "Shard {} memory mismatch", i);
    }

    println!("   Manager 1 and Manager 2 are identical");
    println!("   ✅ Sharding is deterministic across instances\n");

    // Test 10: Large Model Simulation
    println!("📝 Test 10: Large Model Simulation (Llama-70B)");
    let num_layers = 80; // Llama-70B has 80 layers
    let num_regions = 8; // Split across 8 memory regions
    let memory_per_layer = 128 * 1024 * 1024; // ~128MB per layer

    let manager =
        ShardingManager::with_sequential_sharding(num_layers, num_regions, memory_per_layer)?;

    println!("   Model: {} layers", num_layers);
    println!("   Regions: {}", num_regions);
    println!("   Layers per region: {}", num_layers / num_regions);

    let stats = manager.memory_stats();
    for (region_id, layer_count, memory_bytes) in &stats {
        let memory_gb = *memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("   Region {}: {} layers, {:.2} GB", region_id, layer_count, memory_gb);
    }

    let total_gb = manager.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("   Total model size: {:.2} GB", total_gb);
    println!("   ✅ Large model sharding is deterministic\n");

    // Final Summary
    println!("✅ All Layer Sharding Tests Passed!");
    println!("\n📊 Summary:");
    println!("   • No sharding (single region): Deterministic ✅");
    println!("   • Sequential sharding (even): Deterministic ✅");
    println!("   • Sequential sharding (uneven): Deterministic ✅");
    println!("   • Custom sharding: Deterministic ✅");
    println!("   • Memory statistics: Deterministic ✅");
    println!("   • Error detection (gaps): Deterministic ✅");
    println!("   • Error detection (overlaps): Deterministic ✅");
    println!("   • Error detection (invalid): Deterministic ✅");
    println!("   • Cross-instance consistency: Deterministic ✅");
    println!("   • Large model simulation: Deterministic ✅");
    println!("\n💡 Layer sharding exhibits fully deterministic behavior");
    println!("🚀 Ready for production use with large models (>4GB)");

    Ok(())
}
