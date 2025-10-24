/// Chat Model Test
///
/// This test specifically uses the chat variant model with proper chat template formatting
/// to see if we can get coherent output like Ollama.
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 WASM-Chord Chat Model Test");
    println!("=============================\n");

    // Use the chat variant model
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    println!("📂 Loading chat model: {}", model_path);

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

    // === Test Multiple Prompts ===
    let _test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a short poem about the ocean",
    ];

    let config = GenerationConfig {
        max_tokens: 1,    // Just 1 token to test
        temperature: 0.0, // Greedy for deterministic output
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };

    // Test just one prompt first
    let user_prompt = "Hello, how are you?";
    println!("🧪 Testing: {}", user_prompt);
    println!("{}", "=".repeat(50));

    // Format with chat template (same as Ollama)
    let prompt = format!(
        "<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
        user_prompt
    );

    println!("📝 Formatted prompt:\n{}", prompt);

    let start = std::time::Instant::now();
    let generated = model.generate(&prompt, &tokenizer, &config)?;
    let duration = start.elapsed();

    println!("⏱️  Generated in: {:?}", duration);
    println!("🤖 Response:\n{}\n", generated);
    println!("{}", "-".repeat(80));

    // Now let's also test what Ollama produces with the same prompt
    println!("\n🔄 For comparison, here's what Ollama produces:");
    println!("{}", "=".repeat(50));

    Ok(())
}
