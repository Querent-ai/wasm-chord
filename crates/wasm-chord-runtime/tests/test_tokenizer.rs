/// Test tokenizer loading from real GGUF files
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use wasm_chord_core::{GGUFParser, Tokenizer};

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
fn test_tokenizer_from_gguf() {
    let model_path = get_model_path();
    println!("📂 Loading tokenizer from: {}", model_path.display());

    // Parse GGUF
    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);

    let meta = parser.parse_header().expect("Failed to parse header");
    println!("✅ Parsed GGUF metadata");

    // Create tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta).expect("Failed to create tokenizer");
    println!("✅ Tokenizer created");
    println!("   Vocab size: {}", tokenizer.vocab_size());
    println!("   BOS token ID: {}", tokenizer.special_tokens().bos_token_id);
    println!("   EOS token ID: {}", tokenizer.special_tokens().eos_token_id);

    // Test some token lookups
    let bos_str = tokenizer.id_to_token(tokenizer.special_tokens().bos_token_id);
    let eos_str = tokenizer.id_to_token(tokenizer.special_tokens().eos_token_id);
    println!("   BOS token: {:?}", bos_str);
    println!("   EOS token: {:?}", eos_str);

    // Print first 20 tokens
    println!("\n📝 First 20 tokens:");
    for i in 0..20 {
        if let Some(token) = tokenizer.id_to_token(i) {
            println!("   {}: {:?}", i, token);
        }
    }

    assert!(tokenizer.vocab_size() > 0);
}

#[test]
#[ignore]
fn test_tokenizer_encode_decode() {
    let model_path = get_model_path();

    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse header");

    let tokenizer = Tokenizer::from_gguf(&meta).expect("Failed to create tokenizer");

    // Test encoding
    let text = "Hello world";
    let tokens = tokenizer.encode(text, true).expect("Encoding failed");

    println!("🔄 Encode test:");
    println!("   Input: {:?}", text);
    println!("   Tokens: {:?}", tokens);
    println!("   Token count: {}", tokens.len());

    // Test decoding
    let decoded = tokenizer.decode(&tokens, true).expect("Decoding failed");
    println!("   Decoded: {:?}", decoded);

    assert!(!tokens.is_empty());
    assert_eq!(tokens[0], tokenizer.special_tokens().bos_token_id);
}
