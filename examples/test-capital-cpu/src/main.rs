//! CPU-only inference example
//!
//! Tests the model with "What is the capital of France?"
//! Expected output: "Paris"
//!
//! Build and run:
//! ```bash
//! cargo run --release --manifest-path examples/test-capital-cpu/Cargo.toml
//! ```

use anyhow::Result;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<()> {
    println!("🧪 CPU-Only Inference Test");
    println!("===========================\n");

    // Path to model (from env or default)
    let model_path = std::env::var("WASM_CHORD_TEST_MODEL")
        .unwrap_or_else(|_| "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf".to_string());
    println!("📦 Loading model: {}", model_path);

    // Open and parse GGUF file
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    // Extract configuration
    let config_data = parser.extract_config().ok_or_else(|| anyhow::anyhow!("Failed to extract config"))?;
    let config: TransformerConfig = config_data.into();

    println!("✅ Config loaded: {} layers, {} heads", config.num_layers, config.num_heads);

    // Create model
    let mut model = Model::new(config.clone());
    println!("🔧 Model created");

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Load weights
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor in meta.tensors.iter() {
        tensor_loader.register_tensor(tensor.name.clone(), tensor.clone(), tensor.offset);
    }

    // Reopen file for tensor loading
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Weights loaded\n");

    // Initialize GPU if available
    #[cfg(feature = "webgpu")]
    {
        println!("🎮 Initializing WebGPU backend...");
        match model.init_gpu() {
            Ok(_) => println!("✅ WebGPU enabled\n"),
            Err(e) => println!("⚠️  WebGPU init failed (falling back to CPU): {}\n", e),
        }
    }

    // Test prompt
    let prompt = "What is the capital of France?";
    println!("📝 Prompt: \"{}\"", prompt);

    #[cfg(feature = "webgpu")]
    println!("🤖 Generating response (WebGPU-accelerated)...\n");

    #[cfg(not(feature = "webgpu"))]
    println!("🤖 Generating response (CPU-only)...\n");

    // Generation config
    let gen_config = GenerationConfig {
        max_tokens: 20,
        temperature: 0.0, // Greedy for deterministic output
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };

    // Generate
    let response = model.generate(prompt, &tokenizer, &gen_config)?;

    println!("✨ Response: {}", response);
    println!();

    // Check if response contains "Paris"
    if response.to_lowercase().contains("paris") {
        println!("✅ SUCCESS: Model correctly identified Paris as the capital of France!");
    } else {
        println!("⚠️  WARNING: Expected 'Paris' in response, got: {}", response);
    }

    Ok(())
}
