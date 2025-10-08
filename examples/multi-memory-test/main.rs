/// Deterministic Multi-Memory Layout Test
///
/// Tests the multi-memory system with deterministic allocation patterns
/// to ensure consistent behavior across runs.
use wasm_chord_runtime::{MemoryRegion, MultiMemoryLayout};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Multi-Memory Layout Deterministic Test");
    println!("==========================================\n");

    // Test 1: Default Configuration
    println!("📝 Test 1: Default Memory Configuration");
    let layout = MultiMemoryLayout::new();
    let stats = layout.stats();

    println!("   Regions configured: {}", stats.len());
    for (region, used, max, percent) in &stats {
        println!("   {:?}: {} / {} bytes ({:.2}%)", region, used, max, percent);
    }

    // Verify deterministic defaults
    assert_eq!(stats.len(), 4, "Should have exactly 4 memory regions");
    assert_eq!(stats[0].0, MemoryRegion::Weights);
    assert_eq!(stats[1].0, MemoryRegion::Activations);
    assert_eq!(stats[2].0, MemoryRegion::KVCache);
    assert_eq!(stats[3].0, MemoryRegion::Embeddings);
    println!("   ✅ Default configuration is deterministic\n");

    // Test 2: Deterministic Allocation Pattern
    println!("📝 Test 2: Deterministic Allocation Pattern");
    let mut layout = MultiMemoryLayout::new();

    let allocations = vec![
        (MemoryRegion::Weights, 500 * 1024 * 1024, "500MB"),
        (MemoryRegion::Activations, 100 * 1024 * 1024, "100MB"),
        (MemoryRegion::KVCache, 50 * 1024 * 1024, "50MB"),
        (MemoryRegion::Embeddings, 200 * 1024 * 1024, "200MB"),
    ];

    for (region, size, label) in &allocations {
        layout.allocate(*region, *size)?;
        let (used, _max, percent) = layout.region_usage(*region).unwrap();
        println!("   Allocated {} to {:?}: {} bytes ({:.2}%)", label, region, used, percent);
        assert_eq!(used, *size, "Allocated size must match exactly");
    }

    let (total_used, total_max, total_percent) = layout.total_usage();
    println!("   Total: {} / {} bytes ({:.2}%)", total_used, total_max, total_percent);
    assert_eq!(total_used, 850 * 1024 * 1024, "Total should be exactly 850MB");
    println!("   ✅ Allocations are deterministic\n");

    // Test 3: Deallocation Determinism
    println!("📝 Test 3: Deterministic Deallocation");
    layout.deallocate(MemoryRegion::Activations, 50 * 1024 * 1024)?;

    let (used, _max, _) = layout.region_usage(MemoryRegion::Activations).unwrap();
    assert_eq!(used, 50 * 1024 * 1024, "Should have exactly 50MB remaining");
    println!("   Deallocated 50MB from Activations");
    println!("   Remaining: {} bytes", used);
    println!("   ✅ Deallocation is deterministic\n");

    // Test 4: Allocation Limits (Deterministic Failures)
    println!("📝 Test 4: Deterministic Allocation Limits");
    let mut layout = MultiMemoryLayout::new();

    // Embeddings max is 1GB
    let result = layout.allocate(MemoryRegion::Embeddings, 2 * 1024 * 1024 * 1024);
    assert!(result.is_err(), "Should fail when exceeding limit");
    println!("   ✅ Correctly rejected 2GB allocation to Embeddings (max 1GB)");

    // Verify region is unchanged after failed allocation
    let (used, _max, _) = layout.region_usage(MemoryRegion::Embeddings).unwrap();
    assert_eq!(used, 0, "Failed allocation should not change usage");
    println!("   ✅ Failed allocation leaves region unchanged\n");

    // Test 5: Multiple Allocations to Same Region
    println!("📝 Test 5: Cumulative Allocations");
    let mut layout = MultiMemoryLayout::new();

    // Allocate to Weights in multiple steps
    for i in 1..=5 {
        let size = 100 * 1024 * 1024; // 100MB each
        layout.allocate(MemoryRegion::Weights, size)?;
        let (used, _max, _) = layout.region_usage(MemoryRegion::Weights).unwrap();
        let expected = i * size;
        assert_eq!(used, expected, "Step {}: expected {} bytes", i, expected);
        println!("   Step {}: {} bytes allocated", i, used);
    }

    let (final_used, _max, _) = layout.region_usage(MemoryRegion::Weights).unwrap();
    assert_eq!(final_used, 500 * 1024 * 1024, "Should total exactly 500MB");
    println!("   ✅ Cumulative allocations are deterministic\n");

    // Test 6: Reset Determinism
    println!("📝 Test 6: Reset Operation");
    let mut layout = MultiMemoryLayout::new();

    // Allocate to all regions
    layout.allocate(MemoryRegion::Weights, 100 * 1024 * 1024)?;
    layout.allocate(MemoryRegion::Activations, 50 * 1024 * 1024)?;
    layout.allocate(MemoryRegion::KVCache, 25 * 1024 * 1024)?;

    let (before_total, _, _) = layout.total_usage();
    println!("   Before reset: {} bytes", before_total);
    assert!(before_total > 0, "Should have allocations");

    layout.reset();

    let (after_total, _, _) = layout.total_usage();
    println!("   After reset: {} bytes", after_total);
    assert_eq!(after_total, 0, "All allocations should be cleared");
    println!("   ✅ Reset is deterministic\n");

    // Test 7: Cross-Run Consistency
    println!("📝 Test 7: Cross-Run Consistency Check");

    // Create two identical layouts and perform identical operations
    let mut layout1 = MultiMemoryLayout::new();
    let mut layout2 = MultiMemoryLayout::new();

    let operations = vec![
        (MemoryRegion::Weights, 256 * 1024 * 1024),
        (MemoryRegion::Activations, 128 * 1024 * 1024),
        (MemoryRegion::KVCache, 64 * 1024 * 1024),
    ];

    for (region, size) in &operations {
        layout1.allocate(*region, *size)?;
        layout2.allocate(*region, *size)?;
    }

    let stats1 = layout1.stats();
    let stats2 = layout2.stats();

    for (i, ((r1, u1, m1, p1), (r2, u2, m2, p2))) in stats1.iter().zip(stats2.iter()).enumerate() {
        assert_eq!(r1, r2, "Region {} type must match", i);
        assert_eq!(u1, u2, "Region {} used must match", i);
        assert_eq!(m1, m2, "Region {} max must match", i);
        assert_eq!(p1, p2, "Region {} percent must match", i);
    }

    println!("   Layout 1 and Layout 2 are identical");
    println!("   ✅ Operations are deterministic across instances\n");

    // Test 8: Boundary Conditions
    println!("📝 Test 8: Boundary Conditions");
    let mut layout = MultiMemoryLayout::new();

    // Test allocation of 0 bytes
    layout.allocate(MemoryRegion::Weights, 0)?;
    let (used, _max, _) = layout.region_usage(MemoryRegion::Weights).unwrap();
    assert_eq!(used, 0, "0-byte allocation should result in 0 usage");
    println!("   ✅ 0-byte allocation handled correctly");

    // Test allocation of 1 byte
    layout.allocate(MemoryRegion::Weights, 1)?;
    let (used, _max, _) = layout.region_usage(MemoryRegion::Weights).unwrap();
    assert_eq!(used, 1, "1-byte allocation should result in 1 usage");
    println!("   ✅ 1-byte allocation handled correctly");

    // Test deallocation beyond current usage (should saturate at 0)
    layout.deallocate(MemoryRegion::Weights, 1000)?;
    let (used, _max, _) = layout.region_usage(MemoryRegion::Weights).unwrap();
    assert_eq!(used, 0, "Deallocating beyond usage should saturate at 0");
    println!("   ✅ Over-deallocation handled correctly\n");

    // Final Summary
    println!("✅ All Multi-Memory Tests Passed!");
    println!("\n📊 Summary:");
    println!("   • Default configuration: Deterministic ✅");
    println!("   • Allocations: Deterministic ✅");
    println!("   • Deallocations: Deterministic ✅");
    println!("   • Limit enforcement: Deterministic ✅");
    println!("   • Cumulative operations: Deterministic ✅");
    println!("   • Reset operation: Deterministic ✅");
    println!("   • Cross-instance consistency: Deterministic ✅");
    println!("   • Boundary conditions: Deterministic ✅");
    println!("\n💡 Multi-memory layout exhibits fully deterministic behavior");

    Ok(())
}
