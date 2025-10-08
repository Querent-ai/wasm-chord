/// Find Token ID Tool
/// This tool finds the token ID for a specific token text
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Find Token ID Tool");
    println!("===================\n");

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    // Load model
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Find specific tokens
    let tokens_to_find =
        vec!["▁concaten", "▁Yes", "▁Hello", "▁Мос", "Dict", "concaten", "Yes", "Hello"];

    println!("\n🔍 Token ID Lookup:");
    println!("==================");

    for token in tokens_to_find {
        if let Some(token_id) = tokenizer.token_to_id(token) {
            println!("'{:12}' → ID: {}", token, token_id);
        } else {
            println!("'{:12}' → NOT FOUND", token);
        }
    }

    // Also check what token ID 3.47 corresponds to (the top prediction)
    println!("\n🔍 Reverse Lookup:");
    println!("==================");

    // The logit was 3.47, but that's not a token ID - let me check the actual top token
    // Let's find the token ID that corresponds to the highest logit
    println!("Note: The logit value 3.47 is not a token ID - it's the logit score.");
    println!("We need to find which token ID has the highest logit in our model output.");

    Ok(())
}
