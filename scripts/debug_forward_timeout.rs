/// Debug script to test forward pass performance
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{TransformerConfig, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug Forward Pass Performance");
    println!("==================================\n");

    // Model path
    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";
    println!("📂 Loading model: {}", model_path);

    // Load model
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config from GGUF")?;
    let config: TransformerConfig = config_data.into();
    println!("✅ Config: {} layers, {} vocab", config.num_layers, config.vocab_size);

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer: {} tokens", tokenizer.vocab_size());

    // Load weights
    println!("📦 Loading weights...");
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

    // Reopen file for loading
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Weights loaded\n");

    // Test simple forward pass
    println!("🧪 Testing forward pass...");
    let test_tokens = vec![1]; // Just BOS token
    println!("  Input tokens: {:?}", test_tokens);
    
    let start = std::time::Instant::now();
    let logits = model.forward(&test_tokens, 0)?;
    let duration = start.elapsed();
    
    println!("✅ Forward pass completed in {:?}", duration);
    println!("  Output logits length: {}", logits.len());
    println!("  Expected length: {}", config.vocab_size);
    
    if logits.len() == config.vocab_size {
        println!("✅ Logits length matches vocab size");
        
        // Show top 5 logits
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  Top 5 logits:");
        for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
            let token_text = tokenizer.id_to_token(*token_id as u32).unwrap_or("<unknown>");
            println!("    {}: {} ({}) = {:.6}", i+1, token_id, token_text, logit);
        }
    } else {
        println!("❌ Logits length mismatch!");
    }

    Ok(())
}
