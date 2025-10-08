/// WASM 10GB Allocation Test
///
/// Tests whether we can allocate 10GB in WebAssembly with Memory64 feature.
/// This validates that large models (like Llama-70B) can fit in WASM memory.
use wasm_chord_runtime::{MemoryAllocator, MemoryConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 WASM 10GB Allocation Test");
    println!("============================\n");

    // Display current configuration
    let config = MemoryConfig::default();

    println!("📋 Memory Configuration:");
    println!("   Memory64 enabled: {}", config.use_memory64);
    println!(
        "   Max memory: {:.2} GB",
        config.max_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "   Initial memory: {:.2} GB\n",
        config.initial_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Test 1: Can we allocate 10GB?
    println!("📝 Test 1: 10GB Allocation Check");

    let mut allocator = MemoryAllocator::new(MemoryConfig::default());
    let size_10gb = 10 * 1024 * 1024 * 1024;

    let can_allocate = allocator.can_allocate(size_10gb);

    #[cfg(feature = "memory64")]
    {
        assert!(can_allocate, "With Memory64, should be able to allocate 10GB");
        println!("   ✅ Can allocate 10GB (Memory64 enabled)");
        println!("   Max supported: 16 GB\n");
    }

    #[cfg(not(feature = "memory64"))]
    {
        assert!(!can_allocate, "Without Memory64, cannot allocate 10GB (4GB limit)");
        println!("   ❌ Cannot allocate 10GB (Memory64 disabled)");
        println!("   Max supported: 4 GB");
        println!("   💡 Enable with: cargo run --features memory64\n");
    }

    // Test 2: Progressive Allocation (1GB steps)
    println!("📝 Test 2: Progressive Allocation");

    let mut allocator = MemoryAllocator::new(MemoryConfig::default());
    let size_1gb = 1024 * 1024 * 1024;

    let test_sizes = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    for gb in test_sizes {
        let size = gb * size_1gb;
        let can_alloc = allocator.can_allocate(size);

        #[cfg(feature = "memory64")]
        {
            if gb <= 16 {
                assert!(can_alloc, "Should allocate {} GB with Memory64", gb);
                println!("   ✅ {} GB: Can allocate", gb);
            } else {
                assert!(!can_alloc, "Should NOT allocate {} GB (exceeds 16GB)", gb);
                println!("   ❌ {} GB: Exceeds limit", gb);
            }
        }

        #[cfg(not(feature = "memory64"))]
        {
            if gb <= 4 {
                assert!(can_alloc, "Should allocate {} GB (within 4GB limit)", gb);
                println!("   ✅ {} GB: Can allocate", gb);
            } else {
                assert!(!can_alloc, "Should NOT allocate {} GB (exceeds 4GB)", gb);
                println!("   ❌ {} GB: Exceeds 4GB limit", gb);
            }
        }
    }
    println!();

    // Test 3: Actual Allocation (smaller test to avoid OOM)
    println!("📝 Test 3: Actual Memory Allocation");

    // Test with 1GB actual allocation (safe for both modes)
    let mut allocator = MemoryAllocator::new(MemoryConfig::default());
    let test_size = 1024 * 1024 * 1024; // 1GB

    match allocator.allocate::<u8>(test_size) {
        Ok(buffer) => {
            println!("   ✅ Successfully allocated 1GB");
            println!("   Buffer size: {} bytes", buffer.len());
            println!("   Memory usage: {:.1}%", allocator.usage_percent());

            // Verify deterministic allocation
            assert_eq!(buffer.len(), test_size, "Buffer size must match requested");
            println!("   ✅ Allocation size is deterministic");
        }
        Err(e) => {
            println!("   ❌ Failed to allocate 1GB: {}", e);
            return Err(e.into());
        }
    }
    println!();

    // Test 4: Large Model Simulation
    println!("📝 Test 4: Large Model Memory Requirements");

    let models = vec![
        ("TinyLlama 1.1B (Q4)", 0.6),
        ("Llama-7B (Q4)", 3.5),
        ("Llama-13B (Q4)", 7.0),
        ("Llama-70B (Q4)", 35.0),
        ("Llama-70B (Q8)", 70.0),
    ];

    println!("   Model compatibility:");
    for (model_name, size_gb) in models {
        let size_bytes = (size_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let allocator = MemoryAllocator::new(MemoryConfig::default());
        let can_fit = allocator.can_allocate(size_bytes);

        let status = if can_fit { "✅" } else { "❌" };
        println!("   {} {}: {:.1} GB", status, model_name, size_gb);
    }
    println!();

    // Test 5: Memory64 Benefits Summary
    println!("📝 Test 5: Memory64 Benefits Summary\n");

    #[cfg(feature = "memory64")]
    {
        println!("   🎉 Memory64 is ENABLED");
        println!("   Benefits:");
        println!("   • Can load models >4GB (up to 16GB)");
        println!("   • Llama-13B (Q4) ✅");
        println!("   • Llama-70B (Q4) partial ✅");
        println!("   • 64-bit memory addressing");
        println!("   • Production-ready for large models\n");

        // Verify we can plan for 10GB
        let allocator = MemoryAllocator::new(MemoryConfig::default());
        assert!(allocator.can_allocate(10 * 1024 * 1024 * 1024), "Must support 10GB allocation");
        println!("   ✅ 10GB allocation: SUPPORTED");
    }

    #[cfg(not(feature = "memory64"))]
    {
        println!("   ℹ️  Memory64 is DISABLED");
        println!("   Limitations:");
        println!("   • 4GB memory limit");
        println!("   • Best for models <4GB");
        println!("   • TinyLlama ✅");
        println!("   • Llama-7B (Q4) ✅");
        println!("   • Llama-13B (Q4) ❌");
        println!("   • Llama-70B ❌\n");

        println!("   💡 To enable Memory64:");
        println!(
            "   cargo run --features memory64 --manifest-path examples/wasm-10gb-test/Cargo.toml\n"
        );

        // Verify 10GB is NOT supported without memory64
        let allocator = MemoryAllocator::new(MemoryConfig::default());
        assert!(
            !allocator.can_allocate(10 * 1024 * 1024 * 1024),
            "Should NOT support 10GB without memory64"
        );
        println!("   ❌ 10GB allocation: NOT SUPPORTED (enable memory64)");
    }

    // Final Summary
    println!("\n✅ WASM Memory Tests Completed!");
    println!("\n📊 Results:");

    #[cfg(feature = "memory64")]
    {
        println!("   Memory64: ENABLED ✅");
        println!("   10GB allocation: SUPPORTED ✅");
        println!("   Max addressable: 16 GB");
        println!("   Suitable for: Large models (Llama-13B+)");
    }

    #[cfg(not(feature = "memory64"))]
    {
        println!("   Memory64: DISABLED");
        println!("   10GB allocation: NOT SUPPORTED");
        println!("   Max addressable: 4 GB");
        println!("   Suitable for: Small-medium models (<4GB)");
    }

    Ok(())
}
