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
    
    let tokens = [1, 450, 6593, 310, 2834, 338, 24847, 8463, 2319, 12808, 14607, 5391, 12559, 18703, 1295, 6415, 278, 1234, 13791];
    
    for &token_id in &tokens {
        let text = tokenizer.decode(&[token_id], true)?;
        println!("Token {}: '{}'", token_id, text);
    }
    
    Ok(())
}
