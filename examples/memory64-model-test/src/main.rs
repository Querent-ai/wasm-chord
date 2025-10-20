//! Memory64 Model Loading and Generation Test
//!
//! This example tests end-to-end generation with Memory64 integration,
//! demonstrating on-demand layer loading for large models.

use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Memory64 Generation Test");
    println!("===========================\n");

    // Check for model path argument
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        // Default to TinyLlama for testing
        "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf"
    };

    println!("📂 Model path: {}", model_path);

    // Check if model exists
    if !std::path::Path::new(model_path).exists() {
        println!("❌ Model file not found: {}", model_path);
        println!("   Usage: cargo run --release --features memory64 [MODEL_PATH]");
        return Ok(());
    }

    // Load model
    println!("\n📦 Loading model...");
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    println!(
        "✅ Config: {} layers, {} vocab, {} hidden",
        config.num_layers, config.vocab_size, config.hidden_size
    );

    // Load tokenizer
    println!("\n🔤 Loading tokenizer...");
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded ({} tokens)", tokenizer.vocab_size());

    // Load weights with Memory64 support
    println!("\n⚙️  Loading weights (Memory64 will activate for models >3GB)...");
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Reopen file for loading
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model loaded successfully\n");

    // Test generation
    println!("🧪 Testing generation...");
    let prompt = "Hello";
    println!("   Prompt: \"{}\"", prompt);

    let gen_config = GenerationConfig {
        max_tokens: 10,
        temperature: 0.0, // Greedy
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    let start = std::time::Instant::now();
    let response = model.generate(&prompt, &tokenizer, &gen_config)?;
    let duration = start.elapsed();

    println!("\n✅ Generation complete!");
    println!("   ⏱️  Time: {:.2}s", duration.as_secs_f64());
    println!("   📝 Generated: \"{}\"", response.trim());
    println!("   ⚡ Speed: {:.2} tok/s", gen_config.max_tokens as f64 / duration.as_secs_f64());

    println!("\n🎉 Memory64 test complete!");

    Ok(())
}
