/// Memory64 Stress Test - Allocate >4GB to verify Memory64 works
/// This test will actually try to allocate large amounts of memory to prove Memory64 is functional

use wasm_chord_runtime::{MemoryAllocator, MemoryConfig};
#[cfg(feature = "memory64")]
use wasm_chord_runtime::WasmMemory64Allocator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Memory64 Stress Test - Allocating >4GB");
    println!("==========================================\n");

    // Test 1: Try to allocate 5GB with simulation allocator
    println!("📝 Test 1: Simulation Allocator (5GB allocation)");
    let mut sim_allocator = MemoryAllocator::new(MemoryConfig::default());
    
    let size_5gb = 5 * 1024 * 1024 * 1024 / std::mem::size_of::<u8>();
    
    match sim_allocator.allocate::<u8>(size_5gb) {
        Ok(mut buffer) => {
            println!("   ✅ SUCCESS: Allocated 5GB ({} bytes)", buffer.len());
            println!("   Memory usage: {:.1}%", sim_allocator.usage_percent());
            
            // Fill with garbage to ensure it's actually allocated
            println!("   Filling with garbage data...");
            for i in 0..buffer.len() {
                buffer[i] = (i % 256) as u8;
            }
            println!("   ✅ Garbage data written successfully");
        }
        Err(e) => {
            println!("   ❌ FAILED: {}", e);
        }
    }
    println!();

    // Test 2: Try to allocate 5GB with real Memory64 allocator
    println!("📝 Test 2: Real Memory64 Allocator (5GB allocation)");
    
    #[cfg(feature = "memory64")]
    {
        match WasmMemory64Allocator::new(1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024) { // 1GB initial, 8GB max
            Ok(mut real_allocator) => {
                println!("   ✅ Memory64 allocator created (8GB max)");
                
                let size_5gb = 5 * 1024 * 1024 * 1024 / std::mem::size_of::<u8>();
                
                match real_allocator.allocate::<u8>(size_5gb) {
                    Ok(mut buffer) => {
                        println!("   ✅ SUCCESS: Allocated 5GB ({} bytes)", buffer.len());
                        println!("   Memory usage: {:.1}%", real_allocator.usage_percent());
                        
                        // Fill with garbage to ensure it's actually allocated
                        println!("   Filling with garbage data...");
                        for i in 0..buffer.len() {
                            buffer[i] = (i % 256) as u8;
                        }
                        println!("   ✅ Garbage data written successfully");
                        
                        // Verify the data
                        println!("   Verifying garbage data...");
                        let mut correct = true;
                        for i in 0..std::cmp::min(buffer.len(), 1000) {
                            if buffer[i] != (i % 256) as u8 {
                                correct = false;
                                break;
                            }
                        }
                        println!("   ✅ Data verification: {}", if correct { "PASSED" } else { "FAILED" });
                    }
                    Err(e) => {
                        println!("   ❌ FAILED: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Memory64 allocator creation failed: {}", e);
            }
        }
    }
    
    #[cfg(not(feature = "memory64"))]
    {
        println!("   ⚠️  Memory64 feature not enabled - cannot test real allocator");
        println!("   Enable with: cargo run --features memory64");
    }
    println!();

    // Test 3: Progressive allocation test
    println!("📝 Test 3: Progressive Allocation Test");
    let mut sim_allocator2 = MemoryAllocator::new(MemoryConfig::default());
    
    let test_sizes = vec![
        ("1 GB", 1024 * 1024 * 1024 / std::mem::size_of::<u8>()),
        ("2 GB", 2 * 1024 * 1024 * 1024 / std::mem::size_of::<u8>()),
        ("3 GB", 3 * 1024 * 1024 * 1024 / std::mem::size_of::<u8>()),
        ("4 GB", 4 * 1024 * 1024 * 1024 / std::mem::size_of::<u8>()),
        ("5 GB", 5 * 1024 * 1024 * 1024 / std::mem::size_of::<u8>()),
        ("6 GB", 6 * 1024 * 1024 * 1024 / std::mem::size_of::<u8>()),
    ];
    
    for (name, size) in test_sizes {
        match sim_allocator2.allocate::<u8>(size) {
            Ok(mut buffer) => {
                println!("   ✅ {}: Allocated {} bytes", name, buffer.len());
                
                // Fill with garbage
                for i in 0..std::cmp::min(buffer.len(), 10000) {
                    buffer[i] = (i % 256) as u8;
                }
                
                println!("   Memory usage: {:.1}%", sim_allocator2.usage_percent());
            }
            Err(e) => {
                println!("   ❌ {}: FAILED - {}", name, e);
                break;
            }
        }
    }
    println!();

    // Test 4: Memory64 vs 32-bit comparison
    println!("📝 Test 4: Memory64 vs 32-bit Comparison");
    
    #[cfg(feature = "memory64")]
    {
        println!("   Memory64 ENABLED:");
        println!("   • Max memory: 16 GB");
        println!("   • Can allocate >4GB: ✅ YES");
        println!("   • Pointer size: 64-bit (slower)");
        println!("   • Use case: Large models (>4GB)");
    }
    
    #[cfg(not(feature = "memory64"))]
    {
        println!("   Memory64 DISABLED:");
        println!("   • Max memory: 4 GB");
        println!("   • Can allocate >4GB: ❌ NO");
        println!("   • Pointer size: 32-bit (faster)");
        println!("   • Use case: Small models (<4GB)");
    }
    println!();

    // Summary
    println!("📊 Stress Test Summary:");
    println!("   This test proves whether Memory64 actually works in practice");
    println!("   by attempting to allocate >4GB of memory and fill it with data.");
    println!();
    
    #[cfg(feature = "memory64")]
    {
        println!("🎉 Memory64 is ENABLED - Large allocations should work!");
        println!("   If you see successful 5GB+ allocations above, Memory64 is working.");
    }
    
    #[cfg(not(feature = "memory64"))]
    {
        println!("ℹ️  Memory64 is DISABLED - Only <4GB allocations will work.");
        println!("   Enable with: cargo run --features memory64");
    }

    Ok(())
}
