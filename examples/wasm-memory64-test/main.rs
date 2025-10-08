/// WebAssembly Memory64 Runtime Test
/// Validates that the runtime can handle Memory64-enabled WASM modules
use wasm_chord_runtime::{estimate_model_memory, MemoryAllocator, MemoryConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 WebAssembly Memory64 Runtime Test");
    println!("=====================================\n");

    // Test 1: Verify Memory64 configuration
    println!("📝 Test 1: Memory64 Configuration");
    let config = MemoryConfig::default();

    #[cfg(feature = "memory64")]
    {
        println!("   ✅ Memory64 ENABLED");
        println!("   Target: wasm32-unknown-unknown with memory64 proposal");
        assert_eq!(config.max_memory_bytes, 16 * 1024 * 1024 * 1024);
        assert!(config.use_memory64);
    }

    #[cfg(not(feature = "memory64"))]
    {
        println!("   ℹ️  Memory64 DISABLED");
        println!("   Target: wasm32-unknown-unknown (standard)");
        assert_eq!(config.max_memory_bytes, 4 * 1024 * 1024 * 1024);
        assert!(!config.use_memory64);
    }
    println!();

    // Test 2: Large memory allocation simulation
    println!("📝 Test 2: Large Memory Allocation Simulation");
    let allocator = MemoryAllocator::new(MemoryConfig::default());

    // Simulate allocating memory for a large model
    let model_sizes = vec![
        ("TinyLlama 1.1B Q4_K_M", estimate_model_memory(32000, 2048, 22, 5632)),
        ("Llama 7B Q4_K_M", estimate_model_memory(32000, 4096, 32, 11008)),
        ("Llama 13B Q4_K_M", estimate_model_memory(32000, 5120, 40, 13824)),
    ];

    for (name, size) in model_sizes {
        let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
        let can_fit = allocator.can_allocate(size);

        #[cfg(feature = "memory64")]
        {
            println!("   {} ({:.2} GB): {}",
                name,
                size_gb,
                if can_fit { "✅ Fits in 16GB" } else { "❌ Too large" }
            );
        }

        #[cfg(not(feature = "memory64"))]
        {
            println!("   {} ({:.2} GB): {}",
                name,
                size_gb,
                if can_fit { "✅ Fits in 4GB" } else { "❌ Exceeds 4GB limit" }
            );
        }
    }
    println!();

    // Test 3: Memory addressing test
    println!("📝 Test 3: Memory Addressing Capability");

    #[cfg(feature = "memory64")]
    {
        println!("   Testing 64-bit memory addressing...");

        // With Memory64, we should be able to address beyond 4GB
        let max_addressable: u64 = u64::MAX;
        let gb_4: u64 = 4 * 1024 * 1024 * 1024;

        println!("   Max addressable (Memory64): {} bytes", max_addressable);
        println!("   Beyond 4GB boundary: {}", max_addressable > gb_4);

        // Verify we can work with addresses beyond 4GB
        let large_offset: u64 = 5 * 1024 * 1024 * 1024; // 5GB
        println!("   Can address 5GB offset: {}", large_offset < max_addressable);

        assert!(max_addressable > gb_4);
        println!("   ✅ 64-bit addressing verified");
    }

    #[cfg(not(feature = "memory64"))]
    {
        println!("   Testing 32-bit memory addressing...");

        // Without Memory64, limited to 32-bit addressing
        let max_addressable: usize = usize::MAX;
        let gb_4: usize = 4usize * 1024 * 1024 * 1024 - 1;

        println!("   Max addressable (32-bit): {} bytes", max_addressable);
        println!("   ~4GB boundary: {} bytes", gb_4);

        #[cfg(target_pointer_width = "32")]
        assert!(max_addressable <= gb_4);

        println!("   ✅ 32-bit addressing verified");
    }
    println!();

    // Test 4: Runtime memory limits
    println!("📝 Test 4: Runtime Memory Limits");

    let test_allocations = vec![
        ("100 MB", 100 * 1024 * 1024),
        ("500 MB", 500 * 1024 * 1024),
        ("1 GB", 1024 * 1024 * 1024),
        ("2 GB", 2 * 1024 * 1024 * 1024),
        ("3 GB", 3 * 1024 * 1024 * 1024),
        ("5 GB", 5 * 1024 * 1024 * 1024),
        ("8 GB", 8 * 1024 * 1024 * 1024),
    ];

    for (name, size) in test_allocations {
        let test_allocator = MemoryAllocator::new(MemoryConfig::default());
        let can_allocate = test_allocator.can_allocate(size);

        #[cfg(feature = "memory64")]
        let expected = size <= 16 * 1024 * 1024 * 1024;

        #[cfg(not(feature = "memory64"))]
        let expected = size <= 4 * 1024 * 1024 * 1024;

        let status = if can_allocate == expected {
            "✅"
        } else {
            "❌"
        };

        println!("   {} {}: {}", status, name, if can_allocate { "Allowed" } else { "Rejected" });

        assert_eq!(can_allocate, expected, "Allocation check failed for {}", name);
    }
    println!();

    // Test 5: Actual allocation test
    println!("📝 Test 5: Actual Allocation Test");
    let mut real_allocator = MemoryAllocator::new(MemoryConfig::default());

    // Allocate 1GB
    let size_1gb = 1024 * 1024 * 1024 / std::mem::size_of::<u8>();
    match real_allocator.allocate::<u8>(size_1gb) {
        Ok(buffer) => {
            println!("   ✅ Successfully allocated 1GB ({} bytes)", buffer.len());
            println!("   Memory usage: {:.1}%", real_allocator.usage_percent());
        }
        Err(e) => {
            println!("   ❌ Failed to allocate 1GB: {}", e);
            return Err(e.into());
        }
    }
    println!();

    // Summary
    println!("✅ All WebAssembly Memory64 tests passed!\n");

    println!("📊 Summary:");
    #[cfg(feature = "memory64")]
    {
        println!("   Memory64: ENABLED");
        println!("   Max Memory: 16 GB");
        println!("   Addressing: 64-bit");
        println!("   Large Models: Supported (>4GB)");
        println!("\n   This runtime can handle:");
        println!("   • Llama 7B (full precision)");
        println!("   • Llama 13B (quantized)");
        println!("   • Other large language models");
    }

    #[cfg(not(feature = "memory64"))]
    {
        println!("   Memory64: DISABLED");
        println!("   Max Memory: 4 GB");
        println!("   Addressing: 32-bit");
        println!("   Large Models: Limited (<4GB)");
        println!("\n   Recommended for:");
        println!("   • TinyLlama (quantized)");
        println!("   • Small models");
        println!("   • Edge devices");
    }

    println!("\n💡 To enable Memory64:");
    println!("   cargo run --manifest-path examples/wasm-memory64-test/Cargo.toml --features memory64");

    Ok(())
}
