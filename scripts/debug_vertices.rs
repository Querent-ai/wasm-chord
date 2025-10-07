use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/tinyllama-q4km.gguf";
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    
    // Find what token "vertices" is
    let vertices_tokens = tokenizer.encode("vertices", true)?;
    println!("'vertices' tokens: {:?}", vertices_tokens);
    
    // Decode the first token
    if let Some(&token_id) = vertices_tokens.first() {
        let text = tokenizer.decode(&[token_id], true)?;
        println!("Token {}: '{}'", token_id, text);
    }
    
    // Check what "the answer" would be
    let answer_tokens = tokenizer.encode("the answer", true)?;
    println!("'the answer' tokens: {:?}", answer_tokens);
    
    Ok(())
}
