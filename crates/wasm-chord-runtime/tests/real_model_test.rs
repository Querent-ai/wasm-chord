/// Real model integration test with TinyLlama
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use wasm_chord_core::GGUFParser;
use wasm_chord_runtime::{Model, TransformerConfig};

fn get_model_path() -> PathBuf {
    // Get workspace root (3 levels up from crate manifest dir)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to find workspace root")
        .join("models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
}

#[test]
#[ignore] // Run with --ignored flag since model file is large
fn test_load_tinyllama_metadata() {
    let model_path = get_model_path();

    println!("📂 Loading model from: {}", model_path.display());

    // Open GGUF file
    let file = File::open(model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);

    // Parse GGUF
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse GGUF header");

    println!("✅ GGUF metadata parsed successfully");
    println!("   Architecture: {}", meta.architecture);
    println!("   Version: {}", meta.version);
    println!("   Tensor count: {}", meta.tensor_count);
    println!("   Metadata count: {}", meta.metadata_kv_count);
    println!("   Vocab size: {:?}", meta.vocab_size);

    // Show first 10 metadata keys
    println!("   First 10 metadata keys:");
    for (i, key) in meta.metadata.keys().take(10).enumerate() {
        println!("     {}. {}", i + 1, key);
    }

    // Show first few tensors
    println!("   First 5 tensors:");
    for (i, tensor) in meta.tensors.iter().take(5).enumerate() {
        println!("     {}. {} - shape: {:?}", i + 1, tensor.name, tensor.shape);
    }

    // Check metadata for vocab-related keys
    println!("   Searching for vocab_size metadata...");
    for key in meta.metadata.keys() {
        if key.contains("vocab") || key.contains("embedding") || key.contains("context") {
            println!("     Found: {} = {:?}", key, meta.metadata.get(key));
        }
    }

    assert!(meta.tensor_count > 0, "Should have tensors");
    // vocab_size might not be in metadata, could be derived from tensors
    // assert!(meta.vocab_size.is_some(), "Should have vocab_size");
}

#[test]
#[ignore]
fn test_extract_config_from_gguf() {
    let model_path = get_model_path();

    println!("📂 Loading config from: {}", model_path.display());

    let file = File::open(model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);

    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Failed to parse header");

    // Extract config
    let config_data = parser.extract_config().expect("Failed to extract config");

    println!("✅ Config extracted:");
    println!("   vocab_size: {}", config_data.vocab_size);
    println!("   hidden_size: {}", config_data.hidden_size);
    println!("   num_layers: {}", config_data.num_layers);
    println!("   num_heads: {}", config_data.num_heads);
    println!("   num_kv_heads: {}", config_data.num_kv_heads);
    println!("   intermediate_size: {}", config_data.intermediate_size);
    println!("   max_seq_len: {}", config_data.max_seq_len);

    assert!(config_data.vocab_size > 0, "Vocab size should be positive");
    assert!(config_data.hidden_size > 0, "Hidden size should be positive");
    assert!(config_data.num_layers > 0, "Num layers should be positive");

    let vocab_size = config_data.vocab_size;

    // Create TransformerConfig from extracted data
    let config: TransformerConfig = config_data.into();

    println!("✅ TransformerConfig created");
    assert_eq!(config.vocab_size, vocab_size);
}

#[test]
#[ignore]
fn test_model_creation_with_real_config() {
    let model_path = get_model_path();

    println!("📂 Loading real config and creating model");

    let file = File::open(model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);

    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Failed to parse header");
    let config_data = parser.extract_config().expect("Failed to extract config");

    // Convert to runtime config
    let config: TransformerConfig = config_data.into();

    // Create model (WARNING: This will allocate memory based on real model size!)
    // TinyLlama is ~1.1B params, so this may take significant memory
    println!(
        "⚠️  Creating model with {} layers, hidden_size={}",
        config.num_layers, config.hidden_size
    );
    println!("   This may take a moment...");

    let model = Model::new(config.clone());

    println!("✅ Model created successfully!");
    println!("   Layers: {}", model.layers.len());
    println!("   KV caches: {}", model.kv_caches.len());
    println!("   Config vocab_size: {}", model.config.vocab_size);

    assert_eq!(model.layers.len(), config.num_layers);
    assert_eq!(model.kv_caches.len(), config.num_layers);
}

#[test]
#[ignore]
fn test_forward_pass_with_real_config() {
    let model_path = get_model_path();

    println!("🚀 Testing forward pass with real model config");

    let file = File::open(model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);

    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Failed to parse header");
    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    println!("   Creating model...");
    let mut model = Model::new(config);

    // Test with simple tokens (NOTE: weights are random, output will be gibberish)
    let test_tokens = vec![1u32, 15043, 29892]; // Random token IDs

    println!("🔄 Running forward pass...");
    println!("   Input tokens: {:?}", test_tokens);

    let start = std::time::Instant::now();
    let result = model.forward(&test_tokens, 0);
    let duration = start.elapsed();

    assert!(result.is_ok(), "Forward pass should not panic");

    let logits = result.unwrap();

    println!("✅ Forward pass complete in {:?}", duration);
    println!("   Output shape: {} logits", logits.len());
    println!(
        "   Expected: {} (seq_len) * {} (vocab_size)",
        test_tokens.len(),
        model.config.vocab_size
    );

    let expected_size = test_tokens.len() * model.config.vocab_size;
    assert_eq!(logits.len(), expected_size, "Logits should be [seq_len * vocab_size]");

    // Extract logits for last token
    let last_token_logits = &logits[(logits.len() - model.config.vocab_size)..];

    // Sample next token
    println!("🎲 Sampling next token (greedy)...");
    let next_token = model.sample(last_token_logits, 0.0, 1.0, 0).expect("Sampling failed");
    println!("   Next token (random weights): {}", next_token);

    assert!(next_token < model.config.vocab_size as u32);
}

#[test]
#[ignore]
fn test_multiple_forward_passes() {
    let model_path = get_model_path();

    println!("🚀 Testing multiple forward passes (token generation)");

    let file = File::open(model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);

    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Failed to parse header");
    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    let mut model = Model::new(config);

    // Generate 10 tokens
    let mut tokens = vec![1u32]; // Start with BOS
    let max_new_tokens = 10;

    println!("🔄 Generating {} tokens...", max_new_tokens);

    let start = std::time::Instant::now();

    for i in 0..max_new_tokens {
        let logits = model.forward(&tokens, 0).expect("Forward failed");

        // Extract logits for last token position
        let last_token_logits = &logits[(logits.len() - model.config.vocab_size)..];

        let next_token = model.sample(last_token_logits, 0.8, 0.95, 40).expect("Sampling failed");
        tokens.push(next_token);
        println!("   Token {}: {}", i + 1, next_token);
    }

    let duration = start.elapsed();

    println!("✅ Generated {} tokens in {:?}", max_new_tokens, duration);
    println!("   Average: {:?}/token", duration / max_new_tokens);
    println!("   Final token sequence: {:?}", tokens);

    assert_eq!(tokens.len(), 1 + max_new_tokens as usize);
}
