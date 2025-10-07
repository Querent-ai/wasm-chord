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
    
    // Decode the top tokens from our logits
    let top_tokens = [15099, 8315, 22981, 10532, 1912, 6304, 6891, 30115, 17841, 2462];
    
    println!("Top tokens from our model:");
    for &token_id in &top_tokens {
        let text = tokenizer.decode(&[token_id], true)?;
        println!("  Token {}: '{}'", token_id, text);
    }
    
    // Check what "In Hinduism" would be tokenized as
    let hinduism_tokens = tokenizer.encode("In Hinduism", true)?;
    println!("\n'Hinduism' tokens: {:?}", hinduism_tokens);
    
    // Check what "the meaning" would be tokenized as  
    let meaning_tokens = tokenizer.encode("the meaning", true)?;
    println!("'the meaning' tokens: {:?}", meaning_tokens);
    
    Ok(())
}
