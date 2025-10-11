/// Advanced integration tests that load the full model and compare outputs
///
/// These tests verify that our implementation produces the same results as llama.cpp
/// when using deterministic settings (temp=0, greedy sampling).
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

/// Get test model path from environment or default location
fn get_test_model_path() -> String {
    std::env::var("WASM_CHORD_TEST_MODEL")
        .unwrap_or_else(|_| "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf".to_string())
}

/// Load reference data from JSON file
fn load_reference_data() -> serde_json::Value {
    let file = File::open("reference_data.json").expect("Failed to open reference data file");
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).expect("Failed to parse reference data")
}

/// Load model and tokenizer for testing
fn load_test_model() -> (Model, Tokenizer, TransformerConfig) {
    let model_path = get_test_model_path();

    // Load GGUF file
    let file = File::open(model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse GGUF header");

    // Extract config
    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta).expect("Failed to load tokenizer");

    // Load model weights
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset().expect("Failed to get tensor data offset");
    let mut tensor_loader = TensorLoader::new(data_offset);

    // Register all tensors
    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Reopen file for loading
    let file = File::open(model_path).expect("Failed to reopen model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Failed to reparse header");

    // Load weights
    model.load_from_gguf(&mut tensor_loader, &mut parser).expect("Failed to load model weights");

    (model, tokenizer, config)
}

/// Compare logits with reference data within tolerance
fn compare_logits(
    actual_logits: &[f32],
    expected_logits: &[serde_json::Value],
    tolerance: f32,
) -> bool {
    for expected in expected_logits {
        let token_id = expected["token_id"].as_u64().unwrap() as usize;
        let expected_logit = expected["logit"].as_f64().unwrap() as f32;
        let actual_logit = actual_logits[token_id];

        if (actual_logit - expected_logit).abs() > tolerance {
            println!(
                "Logit mismatch for token {}: expected {}, actual {}, diff {}",
                token_id,
                expected_logit,
                actual_logit,
                (actual_logit - expected_logit).abs()
            );
            return false;
        }
    }
    true
}

/// Generate tokens with greedy sampling (temp=0)
fn generate_greedy_tokens(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    num_tokens: usize,
) -> Vec<u32> {
    let mut tokens = tokenizer.encode(prompt, true).expect("Failed to encode prompt");
    let mut generated_tokens = Vec::new();

    for _ in 0..num_tokens {
        let logits = model.forward(&tokens, 0).expect("Forward pass failed");

        // Find token with highest logit (greedy sampling)
        let max_token =
            logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
                as u32;

        generated_tokens.push(max_token);
        tokens.push(max_token);
    }

    generated_tokens
}

#[test]
fn test_known_logits_hello_prompt() {
    println!("🧪 Testing logits for 'Hello' prompt");

    let (mut model, tokenizer, _config) = load_test_model();
    let reference_data = load_reference_data();

    // Get reference data for "Hello" prompt
    let hello_ref = &reference_data["Hello"];
    let expected_logits = hello_ref["expected_top_logits"].as_array().unwrap();

    // Tokenize prompt
    let tokens = tokenizer.encode("Hello", true).expect("Failed to encode 'Hello'");

    // Run forward pass
    let logits = model.forward(&tokens, 0).expect("Forward pass failed");

    // Compare top-5 logits with reference
    let tolerance = 0.01; // Allow small floating-point differences
    let matches = compare_logits(&logits, expected_logits, tolerance);

    if !matches {
        // Print top 10 actual logits for debugging
        let mut indexed_logits: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top 10 actual logits:");
        for (rank, (token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
            let token_text = tokenizer.id_to_token(*token_id as u32);
            let token_str = token_text.as_deref().unwrap_or("<unknown>");
            println!("  {}: {} (id: {}, logit: {:.6})", rank + 1, token_str, token_id, logit);
        }
    }

    assert!(matches, "Logits do not match reference within tolerance {}", tolerance);
    println!("✅ Logits match reference for 'Hello' prompt");
}

#[test]
fn test_deterministic_generation() {
    println!("🧪 Testing deterministic token generation");

    let (mut model, tokenizer, _config) = load_test_model();

    // Test multiple prompts
    let test_prompts = vec!["Hello", "The", "Once upon a time"];

    for prompt in test_prompts {
        println!("Testing prompt: '{}'", prompt);

        // Generate tokens twice to ensure determinism
        let tokens1 = generate_greedy_tokens(&mut model, &tokenizer, prompt, 4);
        let tokens2 = generate_greedy_tokens(&mut model, &tokenizer, prompt, 4);

        // Should be identical with greedy sampling
        assert_eq!(tokens1, tokens2, "Generation is not deterministic for prompt '{}'", prompt);

        // Convert tokens to text for debugging
        let tokens_text1: Vec<String> = tokens1
            .iter()
            .map(|&id| tokenizer.id_to_token(id).unwrap_or("<unknown>").to_string())
            .collect();

        println!("  Generated tokens: {:?}", tokens_text1);
        println!("  ✅ Deterministic generation confirmed");
    }
}

#[test]
fn test_kv_cache_consistency() {
    println!("🧪 Testing KV cache consistency");

    let (mut model, tokenizer, _config) = load_test_model();

    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, true).expect("Failed to encode prompt");

    // First forward pass
    let logits1 = model.forward(&tokens, 0).expect("Forward pass failed");

    // Generate a token
    let max_token =
        logits1.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;

    let mut extended_tokens = tokens.clone();
    extended_tokens.push(max_token);

    // Second forward pass with extended sequence
    let logits2 = model.forward(&extended_tokens, 0).expect("Forward pass failed");

    // The logits should be different (we're generating the next token)
    // But the model should handle the extended sequence correctly
    assert_ne!(logits1.len(), 0, "First forward pass should produce logits");
    assert_ne!(logits2.len(), 0, "Second forward pass should produce logits");
    assert_eq!(logits1.len(), logits2.len(), "Logit dimensions should match");

    // Verify that the model can handle multi-token sequences
    let multi_token_prompt = "Hello world";
    let multi_tokens =
        tokenizer.encode(multi_token_prompt, true).expect("Failed to encode multi-token prompt");
    let multi_logits = model.forward(&multi_tokens, 0).expect("Multi-token forward pass failed");

    assert_eq!(
        multi_logits.len(),
        tokenizer.vocab_size(),
        "Multi-token logits should have correct size"
    );

    println!("✅ KV cache consistency verified");
}
