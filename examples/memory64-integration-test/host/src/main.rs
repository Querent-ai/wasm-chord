//! Memory64 Integration Test - Host Side
//!
//! This example demonstrates setting up the Memory64Runtime on the host side
//! and loading a WASM module that uses Memory64 features.

use anyhow::{anyhow, Context, Result};
use wasm_chord_runtime::memory64_host::{Memory64Runtime, MemoryLayout};
use wasmtime::*;

fn main() -> Result<()> {
    println!("🚀 Memory64 Integration Test - Host Side");
    println!("==========================================\n");

    // Step 1: Create Wasmtime engine with Memory64 support
    println!("📦 Step 1: Creating Wasmtime engine with Memory64 support...");
    let mut config = Config::new();
    config.wasm_memory64(true); // Enable Memory64 proposal
    config.wasm_multi_memory(true); // Enable multi-memory proposal
    let engine = Engine::new(&config)?;
    println!("   ✅ Engine created with Memory64 enabled\n");

    // Step 2: Create Memory64 layout for test
    println!("🧠 Step 2: Creating Memory64 layout...");
    let layout = MemoryLayout::multi(&[
        ("embeddings", 100), // 100MB for embeddings
        ("layer_0", 50),     // 50MB for layer 0
        ("layer_1", 50),     // 50MB for layer 1
        ("lm_head", 50),     // 50MB for LM head
    ])?;
    println!("   📊 Layout created:");
    println!("      - Total regions: {}", layout.regions.len());
    println!("      - Total size: {:.2} MB", layout.total_size as f64 / 1024.0 / 1024.0);
    for region in &layout.regions {
        println!(
            "      - {}: {:.2} MB at offset {}",
            region.name,
            region.size as f64 / 1024.0 / 1024.0,
            region.start_offset
        );
    }
    println!();

    // Step 3: Create Memory64 runtime
    println!("🔧 Step 3: Creating Memory64 runtime...");
    let runtime = Memory64Runtime::new(layout, true);
    println!("   ✅ Runtime created\n");

    // Step 4: Create Wasmtime linker and add Memory64 host functions
    println!("🔗 Step 4: Setting up linker with Memory64 host functions...");
    let mut linker = Linker::new(&engine);
    runtime.add_to_linker(&mut linker)?;
    println!("   ✅ Host functions registered:");
    println!("      - memory64_is_enabled");
    println!("      - memory64_load_layer");
    println!("      - memory64_read");
    println!("      - memory64_get_stats\n");

    // Step 5: Load WASM module
    println!("📂 Step 5: Loading WASM module...");
    // Find workspace root by going up from current directory
    let mut wasm_path = std::env::current_dir()?;
    while !wasm_path.join("Cargo.toml").exists() || !wasm_path.join("crates").exists() {
        if !wasm_path.pop() {
            return Err(anyhow!("Could not find workspace root"));
        }
    }
    wasm_path =
        wasm_path.join("target/wasm32-unknown-unknown/release/memory64_integration_wasm.wasm");

    if !wasm_path.exists() {
        println!("   ⚠️  WASM module not found at: {}", wasm_path.display());
        println!("   ℹ️  Build it first with:");
        println!("      cargo build --release --target wasm32-unknown-unknown -p memory64-integration-wasm");
        return Ok(());
    }

    let module = Module::from_file(&engine, &wasm_path).context("Failed to load WASM module")?;
    println!("   ✅ WASM module loaded\n");

    // Step 6: Create store and instance
    println!("🏪 Step 6: Creating store and instantiating module...");
    let mut store = Store::new(&engine, ());

    // Initialize Memory64 runtime in the store
    runtime.initialize(&mut store)?;
    println!("   ✅ Memory64 initialized in store");

    let instance =
        linker.instantiate(&mut store, &module).context("Failed to instantiate WASM module")?;
    println!("   ✅ WASM module instantiated\n");

    // Step 7: Call WASM test function
    println!("🧪 Step 7: Calling WASM test function...");
    let test_fn = instance
        .get_typed_func::<(), i32>(&mut store, "test_memory64")
        .context("Failed to get test_memory64 function")?;

    let result = test_fn.call(&mut store, ()).context("Failed to call test_memory64 function")?;

    println!("   📊 Test result: {}", result);
    if result == 0 {
        println!("   ✅ Memory64 integration test PASSED!");
    } else {
        println!("   ❌ Memory64 integration test FAILED with code: {}", result);
    }
    println!();

    // Step 8: Load test data into Memory64
    println!("💾 Step 8: Testing data writes to Memory64...");
    let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    runtime.write_model_data(&mut store, 0, &test_data)?;
    println!("   ✅ Wrote {} bytes to Memory64 at offset 0", test_data.len());

    // Register a layer for testing
    runtime.register_layer(&mut store, 0, "test_layer", 0, test_data.len())?;
    println!("   ✅ Registered layer 0 (test_layer) at offset 0, size {}", test_data.len());
    println!();

    // Step 9: Call WASM function to read the data back
    println!("🔄 Step 9: Testing WASM reads from Memory64...");
    let read_test_fn = instance
        .get_typed_func::<(), i32>(&mut store, "test_read_memory64")
        .context("Failed to get test_read_memory64 function")?;

    let read_result =
        read_test_fn.call(&mut store, ()).context("Failed to call test_read_memory64 function")?;

    if read_result == 0 {
        println!("   ✅ WASM successfully read data from Memory64!");
    } else {
        println!("   ❌ WASM read test failed with code: {}", read_result);
    }
    println!();

    // Step 10: Get statistics
    println!("📊 Step 10: Memory64 statistics:");
    let stats = runtime.get_stats(&store)?;
    println!("   - Total reads: {}", stats.reads);
    println!("   - Total writes: {}", stats.writes);
    println!("   - Bytes read: {:.2} MB", stats.bytes_read as f64 / 1024.0 / 1024.0);
    println!("   - Bytes written: {:.2} MB", stats.bytes_written as f64 / 1024.0 / 1024.0);
    println!("   - Errors: {}", stats.errors);
    println!();

    println!("✨ Memory64 integration test completed successfully!");
    println!("==========================================");

    Ok(())
}
