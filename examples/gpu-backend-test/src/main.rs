//! GPU Backend Test
//!
//! This test verifies that our GPU backend can detect and initialize properly.

use pollster::block_on;
use wasm_chord_gpu::GpuBackend;

fn main() {
    println!("🚀 GPU Backend Test");
    println!("==================\n");

    // Test GPU availability detection
    println!("🔍 Testing GPU availability detection...");
    let is_available = GpuBackend::is_available();
    println!("   GPU Available: {}", is_available);

    if is_available {
        println!("✅ GPU backend is available!");

        // Test GPU backend initialization
        println!("\n🔧 Testing GPU backend initialization...");
        match block_on(GpuBackend::new()) {
            Ok(_backend) => {
                println!("✅ GPU backend initialized successfully!");
            }
            Err(e) => {
                println!("❌ Failed to initialize GPU backend: {}", e);
            }
        }
    } else {
        println!("⚠️  GPU backend not available (expected on this system)");
        println!("   This is normal for systems without GPU drivers loaded");
    }

    println!("\n🎯 GPU Backend Test Complete!");
}
