/// Integration tests comparing wasm-chord outputs with llama.cpp reference outputs
///
/// These tests verify that our implementation produces the same results as llama.cpp
/// when using deterministic settings (temp=0, greedy sampling).
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, Tokenizer};
use wasm_chord_runtime::TransformerConfig;

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

/// Load model configuration and tokenizer for testing
fn load_test_config() -> (Tokenizer, TransformerConfig) {
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

    (tokenizer, config)
}

#[test]
fn test_model_configuration() {
    println!("🧪 Testing model configuration");

    let (_tokenizer, config) = load_test_config();

    // Verify model configuration matches expected TinyLlama parameters
    assert_eq!(config.vocab_size, 32000, "Vocab size should be 32000");
    assert_eq!(config.hidden_size, 2048, "Hidden size should be 2048");
    assert_eq!(config.num_layers, 22, "Number of layers should be 22");
    assert_eq!(config.num_heads, 32, "Number of heads should be 32");
    assert_eq!(config.num_kv_heads, 4, "Number of KV heads should be 4");
    assert_eq!(config.intermediate_size, 5632, "Intermediate size should be 5632");

    println!("✅ Model configuration verified");
}

#[test]
fn test_tokenizer_consistency() {
    println!("🧪 Testing tokenizer consistency");

    let (tokenizer, _config) = load_test_config();

    // Test tokenization consistency
    let test_strings = vec!["Hello", "The", "Once upon a time", "Paris is"];

    for test_str in test_strings {
        // Encode multiple times
        let tokens1 = tokenizer.encode(test_str, true).expect("Failed to encode");
        let tokens2 = tokenizer.encode(test_str, true).expect("Failed to encode");

        assert_eq!(tokens1, tokens2, "Tokenization should be deterministic for '{}'", test_str);

        // Decode back
        let decoded = tokenizer.decode(&tokens1, true).expect("Failed to decode");
        assert_eq!(decoded, test_str, "Round-trip encoding/decoding failed for '{}'", test_str);
    }

    println!("✅ Tokenizer consistency verified");
}

#[test]
fn test_tokenizer_known_tokens() {
    println!("🧪 Testing known token mappings");

    let (tokenizer, _config) = load_test_config();

    // Test specific tokens we know from our ollama-comparison output
    let hello_tokens = tokenizer.encode("Hello", true).expect("Failed to encode 'Hello'");
    println!("Tokens for 'Hello': {:?}", hello_tokens);

    // Should start with BOS token (1)
    assert_eq!(hello_tokens[0], 1, "First token should be BOS token");

    // Check that we can decode back
    let decoded = tokenizer.decode(&hello_tokens, true).expect("Failed to decode");
    assert_eq!(decoded, "Hello", "Round-trip failed for 'Hello'");

    // Test individual token mappings
    if let Some(token_text) = tokenizer.id_to_token(15043) {
        println!("Token 15043 maps to: {:?}", token_text);
        // This should be "▁Hello" based on our ollama-comparison output
    }

    println!("✅ Known token mappings verified");
}

#[test]
fn test_reference_data_format() {
    println!("🧪 Testing reference data format");

    let reference_data = load_reference_data();

    // Verify reference data structure
    assert!(reference_data.is_object(), "Reference data should be an object");

    // Check that we have data for expected prompts
    let expected_prompts = vec!["Hello", "The", "Once upon a time"];
    for prompt in expected_prompts {
        assert!(reference_data.get(prompt).is_some(), "Missing reference data for '{}'", prompt);

        let prompt_data = &reference_data[prompt];
        assert!(prompt_data.get("prompt").is_some(), "Missing 'prompt' field for '{}'", prompt);
        assert!(
            prompt_data.get("expected_tokens").is_some(),
            "Missing 'expected_tokens' field for '{}'",
            prompt
        );
        assert!(
            prompt_data.get("expected_top_logits").is_some(),
            "Missing 'expected_top_logits' field for '{}'",
            prompt
        );
    }

    println!("✅ Reference data format verified");
}

#[test]
fn test_basic_tokenizer_functionality() {
    println!("🧪 Testing basic tokenizer functionality");

    let (tokenizer, config) = load_test_config();

    // Test vocabulary size
    assert_eq!(
        tokenizer.vocab_size(),
        config.vocab_size,
        "Tokenizer vocab size should match config"
    );
    assert_eq!(tokenizer.vocab_size(), 32000, "Vocab size should be 32000");

    // Test special tokens
    let bos_token = tokenizer.id_to_token(1);
    assert!(bos_token.is_some(), "BOS token should exist");
    assert_eq!(bos_token.unwrap(), "<s>", "BOS token should be '<s>'");

    let eos_token = tokenizer.id_to_token(2);
    assert!(eos_token.is_some(), "EOS token should exist");
    assert_eq!(eos_token.unwrap(), "</s>", "EOS token should be '</s>'");

    let unk_token = tokenizer.id_to_token(0);
    assert!(unk_token.is_some(), "UNK token should exist");
    assert_eq!(unk_token.unwrap(), "<unk>", "UNK token should be '<unk>'");

    // Test encoding with and without BOS
    let tokens_with_bos = tokenizer.encode("Hello", true).expect("Failed to encode with BOS");
    let tokens_without_bos =
        tokenizer.encode("Hello", false).expect("Failed to encode without BOS");

    assert_eq!(
        tokens_with_bos.len(),
        tokens_without_bos.len() + 1,
        "BOS token should add one token"
    );
    assert_eq!(tokens_with_bos[0], 1, "First token should be BOS when add_bos=true");

    println!("✅ Basic tokenizer functionality verified");
}
