/// End-to-end text generation test
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn get_model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut path = PathBuf::from(manifest_dir);
    path.pop(); // Go to crates/
    path.pop(); // Go to workspace root

    // Try Q8_0 first
    let q8_path = path.join("models/tinyllama-q8.gguf");
    if q8_path.exists() {
        return q8_path;
    }

    let q4km_path = path.join("models/tinyllama-q4km.gguf");
    if q4km_path.exists() {
        q4km_path
    } else {
        path.join("models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
    }
}

#[test]
#[ignore]
fn test_end_to_end_generation() {
    let model_path = get_model_path();
    println!("🚀 End-to-end text generation test");
    println!("📂 Model: {}", model_path.display());

    // === Load Model ===
    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse header");

    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();
    println!("✅ Config loaded: {} layers, {} vocab", config.num_layers, config.vocab_size);

    // === Load Tokenizer ===
    let tokenizer = Tokenizer::from_gguf(&meta).expect("Failed to create tokenizer");
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // === Load Weights ===
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset().expect("Failed to get data offset");
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Reopen file for loading
    let file = File::open(&model_path).expect("Failed to reopen");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Re-parse");

    model.load_from_gguf(&mut tensor_loader, &mut parser).expect("Failed to load weights");
    println!("✅ Weights loaded");

    // === Generate Text ===
    let prompt = "Hello";
    println!("\n🎲 Generating from prompt: {:?}", prompt);

    let config = GenerationConfig { max_tokens: 10, temperature: 0.0, ..Default::default() };

    let start = std::time::Instant::now();
    let generated = model.generate(prompt, &tokenizer, &config).expect("Generation failed");
    let duration = start.elapsed();

    println!("✅ Generation complete in {:?}", duration);
    println!("📝 Result: {:?}", generated);

    // Basic sanity checks
    assert!(!generated.is_empty(), "Generated text should not be empty");
    assert!(generated.len() > prompt.len(), "Generated text should be longer than prompt");

    println!("\n✅ Test passed!");
}

#[test]
#[ignore]
fn test_generation_with_temperature() {
    let model_path = get_model_path();
    println!("🎲 Testing temperature sampling");

    // Load everything (same as above)
    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse header");

    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    let tokenizer = Tokenizer::from_gguf(&meta).expect("Failed to create tokenizer");

    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset().expect("Failed to get data offset");
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let file = File::open(&model_path).expect("Failed to reopen");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Re-parse");

    model.load_from_gguf(&mut tensor_loader, &mut parser).expect("Failed to load weights");

    // Test different temperatures
    let prompt = "The quick brown";

    println!("\nGreedy (temperature=0.0):");
    let greedy_config = GenerationConfig { max_tokens: 5, temperature: 0.0, ..Default::default() };
    let greedy = model.generate(prompt, &tokenizer, &greedy_config).expect("Generation failed");
    println!("  {:?}", greedy);

    println!("\nWith temperature=0.8:");
    let warm_config = GenerationConfig { max_tokens: 5, temperature: 0.8, ..Default::default() };
    let warm = model.generate(prompt, &tokenizer, &warm_config).expect("Generation failed");
    println!("  {:?}", warm);

    assert!(!greedy.is_empty());
    assert!(!warm.is_empty());

    println!("\n✅ Temperature sampling test passed!");
}
