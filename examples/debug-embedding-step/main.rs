/// Debug embedding lookup step by step
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug Embedding Lookup Step by Step");
    println!("=====================================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b.Q4_0.gguf";
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    // Load model
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let mut model = Model::new(config.clone());
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model loaded successfully");

    // Test token 1 (BOS) embedding lookup
    let token_id = 1u32;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;

    println!("\n🧪 Debugging token {} embedding lookup:", token_id);
    println!("  hidden_size = {}, vocab_size = {}", hidden_size, vocab_size);
    println!("  token_embeddings.len() = {}", model.token_embeddings.len());

    // Method 1: Old wrong method (assumes [vocab_size, hidden_size])
    println!("\n📊 Method 1 (OLD WRONG): assumes [vocab_size, hidden_size]");
    let old_start = (token_id as usize) * hidden_size;
    let old_end = old_start + hidden_size;
    println!("  Indices: {} to {}", old_start, old_end);
    if old_end <= model.token_embeddings.len() {
        let old_embedding: Vec<f32> = model.token_embeddings[old_start..old_end].to_vec();
        println!("  First 10 values: {:?}", &old_embedding[..10.min(old_embedding.len())]);
        let sum: f32 = old_embedding.iter().sum();
        println!("  Sum: {:.6}, Mean: {:.6}", sum, sum / hidden_size as f32);
    }

    // Method 2: New correct method (assumes [hidden_size, vocab_size])
    println!("\n📊 Method 2 (NEW CORRECT): assumes [hidden_size, vocab_size]");
    let mut new_embedding = vec![0.0; hidden_size];
    for j in 0..hidden_size {
        let emb_idx = j * vocab_size + (token_id as usize);
        if emb_idx < model.token_embeddings.len() {
            new_embedding[j] = model.token_embeddings[emb_idx];
        }
        if j < 5 {
            println!("  j={}, emb_idx={}, value={:.6}", j, emb_idx, new_embedding[j]);
        }
    }
    println!("  First 10 values: {:?}", &new_embedding[..10.min(new_embedding.len())]);
    let sum: f32 = new_embedding.iter().sum();
    println!("  Sum: {:.6}, Mean: {:.6}", sum, sum / hidden_size as f32);

    // Check what the forward function is actually doing
    println!("\n📊 What forward function actually produces:");
    let test_tokens = vec![token_id];
    match model.forward(&test_tokens, 0) {
        Ok(logits) => {
            // Find top 5 tokens
            let mut indexed_logits: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("Top 5 predicted tokens:");
            for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
                println!("  {}: token {} = {:.3}", i + 1, token_id, logit);
            }
        }
        Err(e) => {
            println!("❌ Forward pass failed: {}", e);
        }
    }

    Ok(())
}
