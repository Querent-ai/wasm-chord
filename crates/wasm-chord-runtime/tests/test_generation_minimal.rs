/// Minimal generation test with debug output
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn get_model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut path = PathBuf::from(manifest_dir);
    path.pop();
    path.pop();

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
fn test_minimal_generation() {
    let model_path = get_model_path();
    println!("🔬 Minimal generation test with detailed logging");
    println!("📂 Model: {}", model_path.display());

    // Load model
    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse header");

    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    println!(
        "📊 Config: {} layers, {} heads, {} kv_heads, head_dim={}",
        config.num_layers,
        config.num_heads,
        config.num_kv_heads,
        config.hidden_size / config.num_heads
    );

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
    println!("✅ Model loaded\n");

    // Test with simple prompt
    let prompt = "Hello";
    println!("📝 Prompt: {:?}", prompt);

    // Check KV cache state before generation
    println!("🗄️  KV cache state before generation:");
    for (i, cache) in model.kv_caches.iter().enumerate() {
        println!("  Layer {}: seq_pos={}, max_size={}", i, cache.seq_pos, cache.max_size);
    }

    // Encode
    let tokens = tokenizer.encode(prompt, true).expect("Failed to encode");
    println!("🔢 Encoded to {} tokens: {:?}", tokens.len(), tokens);

    // Check KV cache after clear (should be in generate())
    // We can't access internal state easily, so let's just run generate

    let result = model.generate(prompt, &tokenizer, 10, 0.0, 1.0, 0).expect("Generation failed");
    println!("\n📝 Generated: {:?}", result);

    // Check if it's repetitive
    if result.contains("automatisch automatisch") {
        println!("❌ FAIL: Repetitive generation detected!");
        println!("   This confirms the bug is still present.");
    } else {
        println!("✅ SUCCESS: Non-repetitive generation!");
    }
}
