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
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 WASM-Chord Simple Text Generation");
    println!("=====================================\n");

    // Model path
    let model_path = "models/tinyllama-q8.gguf";
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
    let max_tokens = 3; // Just a few tokens for comparison
    let temperature = 0.0; // Deterministic greedy sampling

    println!("🎲 Generating text...");
    println!("   Prompt: {:?}", prompt);
    println!("   Max tokens: {}", max_tokens);
    println!("   Temperature: {}", temperature);

    let start = std::time::Instant::now();
    let generated = model.generate(prompt, &tokenizer, max_tokens, temperature, 1.0, 0)?;
    let duration = start.elapsed();

    println!("\n✅ Generation complete in {:?}", duration);
    println!("📝 Result:\n   {}\n", generated);

    Ok(())
}
