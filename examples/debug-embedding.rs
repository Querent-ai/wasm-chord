/// Debug tool to check if token embeddings are correct
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug: Token Embedding Lookup");
    println!("="*60);

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    // Load model
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;

    // Load model weights
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Reopen for loading
    let file = File::open(model_path)?;
    let mut reader = BufReader::new(file);
    model.load_weights(&mut tensor_loader, &mut reader, &mut parser)?;

    // Test: tokenize "Hello" and get its embedding
    let tokens = tokenizer.encode("Hello", false)?;
    println!("\n📝 Tokenized 'Hello' -> {:?}", tokens);

    let token_id = tokens[0];
    println!("   Using token ID: {}", token_id);

    // Run forward pass
    println!("\n🔄 Running forward pass...");
    let logits = model.forward(&tokens, 0)?;

    println!("✅ Forward pass completed");
    println!("   Logits shape: {} (should be vocab_size={})", logits.len(), config.vocab_size);

    // Find top 10 tokens
    let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 10 predicted tokens:");
    for (rank, (token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id as u32).unwrap_or("<unknown>".to_string());
        println!("   {}: '{}' (id={}, logit={:.6})", rank + 1, token_text, token_id, logit);
    }

    // Check what Ollama would predict
    println!("\n💡 Expected (from Ollama): 'As' or ' you' or similar conversational token");
    println!("   Actual top token: '{}'",
        tokenizer.id_to_token(indexed_logits[0].0 as u32).unwrap_or("<unknown>".to_string()));

    if indexed_logits[0].0 == 10945 {
        println!("\n❌ BUG CONFIRMED: Still predicting 'global' (token 10945)");
    }

    Ok(())
}
