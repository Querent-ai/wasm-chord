/// Debug Generation Test
///
/// This test adds extensive debug logging to identify where the generation hangs.
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🐛 Debug Generation Test");
    println!("=======================\n");

    // Enable debug logging
    std::env::set_var("DEBUG", "1");
    std::env::set_var("DEBUG_KV", "1");
    std::env::set_var("DEBUG_LOGITS", "1");

    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
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

    // === Test Simple Generation ===
    let prompt = "Hello";
    let config = GenerationConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };

    println!("🧪 Testing generation with prompt: \"{}\"", prompt);
    println!("📋 Config: {:?}", config);
    println!("{}", "=".repeat(50));

    let start = std::time::Instant::now();

    println!("🚀 Starting generation...");
    match model.generate(prompt, &tokenizer, &config) {
        Ok(generated) => {
            let duration = start.elapsed();
            println!("✅ Generation completed in {:?}", duration);
            println!("📝 Result: \"{}\"", generated);
        }
        Err(e) => {
            let duration = start.elapsed();
            println!("❌ Generation failed in {:?}: {}", duration, e);
        }
    }

    Ok(())
}
