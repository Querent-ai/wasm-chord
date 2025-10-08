/// Memory64 Support Test
/// Tests that Memory64 feature enables larger memory limits
use wasm_chord_runtime::{estimate_model_memory, requires_memory64, MemoryAllocator, MemoryConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Memory64 Support Test");
    println!("========================\n");

    // Test 1: Check Memory64 feature status
    println!("📝 Test 1: Memory64 Feature Status");
    let config = MemoryConfig::default();
    let allocator = MemoryAllocator::new(config);

    #[cfg(feature = "memory64")]
    {
        println!("   ✅ Memory64 feature is ENABLED");
        assert!(allocator.is_memory64_enabled());
        assert_eq!(allocator.max_bytes(), 16 * 1024 * 1024 * 1024);
        println!("   Max memory: 16 GB");
    }

    #[cfg(not(feature = "memory64"))]
    {
        println!("   ℹ️  Memory64 feature is DISABLED");
        assert!(!allocator.is_memory64_enabled());
        assert_eq!(allocator.max_bytes(), 4 * 1024 * 1024 * 1024);
        println!("   Max memory: 4 GB");
    }
    println!();

    // Test 2: Memory allocation within limits
    println!("📝 Test 2: Memory Allocation");
    let mut allocator = MemoryAllocator::new(MemoryConfig::default());

    // Allocate 100MB
    let size_100mb = 100 * 1024 * 1024 / std::mem::size_of::<f32>();
    let buffer = allocator.allocate::<f32>(size_100mb)?;
    println!("   ✓ Allocated 100 MB");
    assert_eq!(buffer.len(), size_100mb);
    assert_eq!(allocator.allocated_bytes(), 100 * 1024 * 1024);
    println!("   ✓ Allocation tracking correct");
    println!();

    // Test 3: Model size estimation
    println!("📝 Test 3: Model Size Estimation");

    // TinyLlama 1.1B
    let tiny_llama = estimate_model_memory(32000, 2048, 22, 5632);
    println!("   TinyLlama 1.1B: {:.2} GB", tiny_llama as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Requires Memory64: {}", requires_memory64(tiny_llama));

    // Llama 7B
    let llama_7b = estimate_model_memory(32000, 4096, 32, 11008);
    println!("   Llama 7B: {:.2} GB", llama_7b as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Requires Memory64: {}", requires_memory64(llama_7b));

    // Llama 13B
    let llama_13b = estimate_model_memory(32000, 5120, 40, 13824);
    println!("   Llama 13B: {:.2} GB", llama_13b as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Requires Memory64: {}", requires_memory64(llama_13b));
    println!();

    // Test 4: Large allocation (only with Memory64)
    println!("📝 Test 4: Large Allocation Test");
    #[cfg(feature = "memory64")]
    {
        println!("   Testing 5GB allocation (requires Memory64)...");
        let mut large_allocator = MemoryAllocator::new(MemoryConfig::default());
        let size_5gb = 5 * 1024 * 1024 * 1024 / std::mem::size_of::<f32>();

        // This should succeed with Memory64
        let result = large_allocator.allocate::<f32>(size_5gb);
        assert!(result.is_ok(), "5GB allocation should succeed with Memory64");
        println!("   ✅ 5GB allocation successful");
        println!("   Memory usage: {:.1}%", large_allocator.usage_percent());
    }

    #[cfg(not(feature = "memory64"))]
    {
        println!("   Testing 5GB allocation (without Memory64)...");
        let mut small_allocator = MemoryAllocator::new(MemoryConfig::default());
        let size_5gb = 5 * 1024 * 1024 * 1024 / std::mem::size_of::<f32>();

        // This should fail without Memory64 (exceeds 4GB limit)
        let result = small_allocator.allocate::<f32>(size_5gb);
        assert!(result.is_err(), "5GB allocation should fail without Memory64");
        println!("   ✅ 5GB allocation correctly rejected (exceeds 4GB limit)");
    }
    println!();

    // Test 5: Memory usage tracking
    println!("📝 Test 5: Memory Usage Tracking");
    let mut tracker = MemoryAllocator::new(MemoryConfig::default());

    let size_1gb = 1024 * 1024 * 1024 / std::mem::size_of::<f32>();
    tracker.allocate::<f32>(size_1gb)?;
    println!("   Allocated: {:.2} GB", tracker.allocated_bytes() as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Usage: {:.1}%", tracker.usage_percent());

    let size_500mb = 500 * 1024 * 1024 / std::mem::size_of::<f32>();
    tracker.allocate::<f32>(size_500mb)?;
    println!("   After +500MB: {:.2} GB ({:.1}%)",
             tracker.allocated_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
             tracker.usage_percent());

    tracker.deallocate(500 * 1024 * 1024);
    println!("   After deallocate: {:.2} GB ({:.1}%)",
             tracker.allocated_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
             tracker.usage_percent());
    println!();

    // Summary
    println!("✅ All Memory64 tests passed!");
    println!("\n📊 Summary:");
    #[cfg(feature = "memory64")]
    println!("   Memory64: ENABLED - Supports models up to 16GB");
    #[cfg(not(feature = "memory64"))]
    println!("   Memory64: DISABLED - Limited to 4GB");

    println!("\n💡 To enable Memory64 for large models (>4GB):");
    println!("   cargo run --manifest-path examples/memory64-test/Cargo.toml --features memory64");

    Ok(())
}
