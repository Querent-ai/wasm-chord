use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 KV Cache Indexing Verification Test");
    println!("=====================================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    let file = std::fs::File::open(model_path)?;
    let mut parser = GGUFParser::new(file);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    let mut model = Model::new(config.clone());
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;

    let tokenizer = Tokenizer::from_gguf(&meta)?;

    println!("📊 Model Configuration:");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_heads: {}", config.num_heads);
    println!("  num_kv_heads: {}", config.num_kv_heads);
    println!("  head_dim: {}", config.hidden_size / config.num_heads);
    println!("  num_queries_per_kv: {}", config.num_heads / config.num_kv_heads);

    // Test with a simple prompt
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false)?;
    println!("\n🎯 Testing with prompt: \"{}\"", prompt);
    println!("  Tokens: {:?}", tokens);

    // Enable KV cache debugging
    std::env::set_var("DEBUG_KV", "1");
    std::env::set_var("DEBUG_KV_DETAILED", "1");

    // Test forward pass
    match model.forward(&tokens, tokens.len() - 1) {
        Ok(logits) => {
            println!("\n✅ Forward pass successful!");
            println!("  Logits length: {}", logits.len());

            // Find top 5 tokens
            let mut indexed_logits: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("  Top 5 predicted tokens:");
            for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
                match tokenizer.decode(&[*token_id as u32], true) {
                    Ok(txt) => println!("    {}: {} = {:.3} ({:?})", i + 1, token_id, logit, txt),
                    Err(_) => {
                        println!("    {}: {} = {:.3} (<decode-fail>)", i + 1, token_id, logit)
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Forward pass failed: {}", e);
        }
    }

    Ok(())
}
