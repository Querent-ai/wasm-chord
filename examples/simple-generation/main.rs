/// Simple text generation example
///
/// This example demonstrates end-to-end text generation:
/// 1. Load GGUF model and tokenizer
/// 2. Generate text from a prompt
/// 3. Display the result
///
/// Run with: cargo run --release --example simple_generation
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 WASM-Chord Simple Text Generation");
    println!("=====================================\n");

    // Model path - use Q4_K model to match llama.cpp test
    let model_path = "models/tinyllama-1.1b.Q4_K_M.gguf"; // Base model, higher quality
    println!("📂 Loading model: {}", model_path);

    // === Load Model ===
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config from GGUF")?;
    let config: TransformerConfig = config_data.into();
    println!("✅ Config: {} layers, {} vocab", config.num_layers, config.vocab_size);

    // === Load Tokenizer ===
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer: {} tokens", tokenizer.vocab_size());

    // === Load Weights ===
    println!("📦 Loading weights...");
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
    println!("✅ Weights loaded\n");

    // === Generate Text ===
    let prompt = "Hello";

    let config = GenerationConfig {
        max_tokens: 5,    // Generate 5 tokens
        temperature: 0.0, // Deterministic output
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    // Enable debug mode to see what's happening
    // std::env::set_var("DEBUG_FORWARD", "1");
    // std::env::set_var("DEBUG_KV", "1");

    // Debug: Check tokenization
    let tokens = tokenizer.encode(prompt, false)?;
    println!("🔍 Tokenization:");
    println!("   Input: {:?}", prompt);
    println!("   Tokens: {:?}", tokens);
    println!("   Count: {}", tokens.len());

    println!("\n🎲 Generating text...");
    println!("   Prompt: {:?}", prompt);
    println!("   Config: {:?}", config);

    let start = std::time::Instant::now();
    let generated = model.generate(&prompt, &tokenizer, &config)?;
    let duration = start.elapsed();

    println!("\n✅ Generation complete in {:?}", duration);
    println!("📝 Result:\n   {}\n", generated);

    Ok(())
}
