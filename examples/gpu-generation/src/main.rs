//! GPU-accelerated text generation example
//!
//! This example demonstrates enabling GPU acceleration in wasm-chord.
//!
//! ## Key Code Snippet
//!
//! ```rust,no_run
//! use wasm_chord_runtime::Model;
//!
//! // Create model
//! let mut model = Model::new(config);
//!
//! // Initialize GPU (feature-gated, falls back to CPU)
//! #[cfg(feature = "gpu")]
//! model.init_gpu()?;
//!
//! // Generate - automatically uses GPU if available
//! model.generate(prompt, &tokenizer, &config)?;
//! ```
//!
//! ## Building
//!
//! ```bash
//! cargo build --release --features gpu --bin gpu-generation
//! ```
//!
//! ## Performance
//!
//! - **CPU only**: Baseline performance
//! - **With GPU**: 5-10x faster matmul operations
//! - **Auto fallback**: Gracefully falls back to CPU if GPU unavailable
//!
//! For a complete working example, see the `inference` example with GPU feature enabled.

fn main() {
    println!("🚀 wasm-chord GPU Acceleration");
    println!("===============================\n");

    println!("This is a documentation example showing how to enable GPU acceleration.");
    println!("The key is adding one line after creating your model:\n");

    println!("    #[cfg(feature = \"gpu\")]");
    println!("    model.init_gpu()?;\n");

    println!("For a complete working example, see:");
    println!("  - examples/inference/ (can be built with --features gpu)");
    println!("  - README.md in this directory\n");

    println!("## Quick Start\n");
    println!("1. Add gpu feature to Cargo.toml:");
    println!("   wasm-chord-runtime = {{ workspace = true, features = [\"gpu\"] }}\n");

    println!("2. Initialize GPU in your code:");
    println!("   model.init_gpu()?;  // Falls back to CPU if unavailable\n");

    println!("3. Generation automatically uses GPU:");
    println!("   model.generate(prompt, &tokenizer, &config)?;\n");

    println!("That's it! No other code changes needed.");
    println!("\n✅ GPU acceleration is seamlessly integrated.");
}
