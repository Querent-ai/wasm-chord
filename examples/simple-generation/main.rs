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
use wasm_chord_runtime::{ChatMessage, ChatTemplate, GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 WASM-Chord Simple Text Generation");
    println!("=====================================\n");

    // Model path - use Q4_K model to match llama.cpp test
    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf"; // Base model, higher quality
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

    // === Initialize GPU (if available) ===
    #[cfg(feature = "gpu")]
    {
        println!("🎮 Initializing GPU...");
        match model.init_gpu() {
            Ok(()) => println!("✅ GPU initialized successfully!"),
            Err(e) => println!("⚠️  GPU unavailable, using CPU: {}", e),
        }
    }

    // === Test with Chat Template ===
    println!("\n📋 Testing with chat template...");
    let template = ChatTemplate::ChatML;
    let conversation = vec![
        ChatMessage::system("You are a helpful AI assistant."),
        ChatMessage::user("What is 2+2?"),
    ];
    let chat_prompt = template.format(&conversation)?;

    let config = GenerationConfig {
        max_tokens: 1,    // Just generate 1 token to test
        temperature: 0.0, // Deterministic/greedy
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    println!("🔍 Chat prompt:\n{}\n", chat_prompt);

    let start = std::time::Instant::now();
    let response = model.generate(&chat_prompt, &tokenizer, &config)?;
    let duration = start.elapsed();

    // Extract just the assistant response
    let assistant_response = if let Some(idx) = response.rfind("<|assistant|>") {
        response[idx + 13..].trim()
    } else {
        response.trim()
    };

    println!("\n✅ Generation complete in {:?}", duration);
    println!("📝 Assistant response:\n   {}\n", assistant_response);

    Ok(())
}
