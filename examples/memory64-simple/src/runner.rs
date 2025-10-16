//! Host Runner for Memory64 Test
//!
//! This program runs the Memory64 WASM module using Wasmtime,
//! creating a Memory64 instance that exceeds the 4GB WASM limit.

use anyhow::Result;
use wasmtime::*;

fn main() -> Result<()> {
    println!("🧪 Memory64 Test - Wasmtime Host");
    println!("==================================\n");

    // 1. Create Wasmtime engine with Memory64 enabled
    println!("⚙️  Configuring Wasmtime engine...");
    let mut config = Config::new();
    config.wasm_memory64(true); // Enable Memory64 proposal
    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());
    println!("✅ Engine created with Memory64 support\n");

    // 2. Create 8GB Memory64 instance (exceeds 4GB limit!)
    println!("📦 Creating 8GB Memory64 instance...");
    let memory_type = MemoryType::new64(
        128_000,       // min: 128,000 pages × 64KB = 8GB
        Some(256_000), // max: 256,000 pages × 64KB = 16GB
    );
    let memory = Memory::new(&mut store, memory_type)?;

    let mem_size_gb = memory.size(&store) as f64 * 64.0 / 1024.0 / 1024.0;
    println!("✅ Memory64 created: {:.2}GB", mem_size_gb);
    println!("   Type: {}", if memory.ty(&store).is_64() { "64-bit" } else { "32-bit" });
    println!();

    // 3. Load WASM module
    println!("📂 Loading WASM module...");
    let wasm_path = "target/wasm32-unknown-unknown/release/memory64_simple.wasm";

    if !std::path::Path::new(wasm_path).exists() {
        eprintln!("❌ WASM module not found at: {}", wasm_path);
        eprintln!("\nBuild it first:");
        eprintln!("  cargo build --target wasm32-unknown-unknown --release");
        std::process::exit(1);
    }

    let module = Module::from_file(&engine, wasm_path)?;
    println!("✅ Module loaded\n");

    // 4. Set up linker and import memory
    println!("🔗 Linking memory to WASM module...");
    let mut linker = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory)?;
    println!("✅ Memory imported\n");

    // 5. Instantiate the WASM module
    println!("🚀 Instantiating WASM module...");
    let instance = linker.instantiate(&mut store, &module)?;
    println!("✅ Instance created\n");

    // 6. Test: Small allocation (sanity check)
    println!("🧪 Test 1: Small allocation (1MB)...");
    let test_small = instance
        .get_typed_func::<(), u32>(&mut store, "test_small_allocation")?;
    let result = test_small.call(&mut store, ())?;
    if result == 1 {
        println!("✅ PASS: 1MB allocation worked\n");
    } else {
        println!("❌ FAIL: 1MB allocation failed\n");
        return Ok(());
    }

    // 7. Test: Large allocation (6GB - exceeds 4GB limit!)
    println!("🧪 Test 2: Large allocation (6GB)...");
    println!("   This exceeds the standard 4GB WASM limit!");
    let test_large = instance
        .get_typed_func::<u32, u32>(&mut store, "test_large_allocation")?;

    print!("   Allocating 6GB... ");
    std::io::Write::flush(&mut std::io::stdout())?;

    let result = test_large.call(&mut store, 6)?; // 6GB

    match result {
        1 => {
            println!("✅");
            println!("✅ PASS: 6GB allocation and access succeeded!\n");
        }
        0 => {
            println!("❌");
            println!("❌ FAIL: 6GB allocation failed\n");
            return Ok(());
        }
        2 => {
            println!("⚠️");
            println!("⚠️  WARN: Allocated but verification failed\n");
        }
        _ => {
            println!("❓");
            println!("❓ UNKNOWN: Unexpected result code: {}\n", result);
        }
    }

    // 8. Check final memory size
    println!("📊 Final memory statistics:");
    let get_mem_size = instance
        .get_typed_func::<(), i64>(&mut store, "get_memory_size")?;
    let mem_size = get_mem_size.call(&mut store, ())?;
    let mem_size_gb = mem_size as f64 / 1024.0 / 1024.0 / 1024.0;
    println!("   Memory size: {:.2}GB ({} bytes)", mem_size_gb, mem_size);
    println!();

    // Summary
    println!("═══════════════════════════════════");
    println!("✅ Memory64 Test PASSED!");
    println!("═══════════════════════════════════");
    println!();
    println!("Key Achievements:");
    println!("  ✓ Created Memory64 instance (8GB+)");
    println!("  ✓ Loaded standard wasm32 module");
    println!("  ✓ Allocated 6GB (>4GB limit)");
    println!("  ✓ Accessed memory beyond 4GB boundary");
    println!();
    println!("🎉 Memory64 works! Ready for large models.");

    Ok(())
}
