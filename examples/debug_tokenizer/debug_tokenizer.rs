/// Debug tokenizer and sampling issues
/// 
/// This test will help identify why we're getting Unicode characters
/// instead of coherent English text like Ollama.

use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig, GenerationConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Tokenizer and Sampling Debug");
    println!("==============================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b.Q4_K_M.gguf";
    println!("📂 Loading model: {}", model_path);
    
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded with {} tokens", tokenizer.vocab_size());
    
    // Test tokenization
    let test_prompt = "The weather today is";
    println!("\n🔤 Testing tokenization:");
    println!("   Prompt: \"{}\"", test_prompt);
    
    let tokens = tokenizer.encode(test_prompt, true)?;
    println!("   Encoded tokens: {:?}", tokens);
    
    // Decode back to see what we get
    let decoded = tokenizer.decode(&tokens, true)?;
    println!("   Decoded back: \"{}\"", decoded);
    
    // Check what some high-probability tokens decode to
    println!("\n🎯 Checking high-probability tokens:");
    let test_tokens = [10667, 29330, 29705, 10335, 6972];
    for token_id in test_tokens {
        if let Some(token_str) = tokenizer.id_to_token(token_id) {
            println!("   Token {}: \"{}\"", token_id, token_str);
        }
    }
    
    // Check some common English words
    println!("\n📝 Checking common English words:");
    let common_words = ["the", "weather", "today", "is", "good", "bad", "sunny", "rainy"];
    for word in common_words {
        if let Some(token_id) = tokenizer.token_to_id(word) {
            println!("   \"{}\" -> token {}", word, token_id);
        } else {
            println!("   \"{}\" -> NOT FOUND", word);
        }
    }
    
    // Check vocabulary structure
    println!("\n📚 Vocabulary sample (first 20 tokens):");
    for i in 0..20.min(tokenizer.vocab_size()) {
        if let Some(token_str) = tokenizer.id_to_token(i as u32) {
            println!("   Token {}: \"{}\"", i, token_str);
        }
    }
    
    Ok(())
}
