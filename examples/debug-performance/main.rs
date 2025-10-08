/// Performance Debug Tool
/// This tool identifies performance bottlenecks and matrix orientation issues

use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Performance Debug Tool");
    println!("=========================\n");

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";
    
    // Load model
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    let mut model = Model::new(config.clone());
    
    // Load weights
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model loaded successfully");

    // Test token: "Hello"
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false)?;
    println!("🔍 Testing with prompt: '{}' -> {:?}", prompt, tokens);

    // Test 1: Single token forward pass timing
    println!("\n--- Test 1: Single Token Forward Pass ---");
    let start = Instant::now();
    let logits = model.forward(&tokens, 0)?;
    let duration = start.elapsed();
    println!("   Single forward pass took: {:?}", duration);
    println!("   Logits shape: {} (expected: {})", logits.len(), config.vocab_size);

    // Test 2: Multiple tokens forward pass timing
    println!("\n--- Test 2: Multiple Tokens Forward Pass ---");
    let multi_tokens = vec![tokens[0], tokens[0]]; // Same token twice
    let start = Instant::now();
    let logits = model.forward(&multi_tokens, 0)?;
    let duration = start.elapsed();
    println!("   Multi-token forward pass took: {:?}", duration);
    println!("   Logits shape: {} (expected: {})", logits.len(), multi_tokens.len() * config.vocab_size);

    // Test 3: Check matrix orientation by comparing manual vs model computation
    println!("\n--- Test 3: Matrix Orientation Check ---");
    
    // Get token embedding manually
    let token_id = tokens[0] as usize;
    let mut manual_embedding = vec![0.0; config.hidden_size];
    let token_embeddings = &model.token_embeddings;
    
    for i in 0..config.hidden_size {
        let emb_idx = i * config.vocab_size + token_id;
        if emb_idx < token_embeddings.len() {
            manual_embedding[i] = token_embeddings[emb_idx];
        }
    }
    
    // Get model's embedding
    let model_embedding = model.forward(&[tokens[0]], 0)?;
    let model_embedding = &model_embedding[..config.hidden_size];
    
    // Compare
    let mut max_diff = 0.0;
    for i in 0..config.hidden_size {
        let diff = (manual_embedding[i] - model_embedding[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    println!("   Manual embedding vs model embedding max diff: {:.6}", max_diff);
    if max_diff > 1e-5 {
        println!("   ⚠️  WARNING: Large difference suggests matrix orientation issue!");
    } else {
        println!("   ✅ Matrix orientation appears correct");
    }

    // Test 4: Check LM head orientation
    println!("\n--- Test 4: LM Head Orientation Check ---");
    
    // Manual LM head computation
    let lm_head_weights = &model.lm_head;
    let mut manual_logits = vec![0.0; config.vocab_size];
    
    for v_idx in 0..config.vocab_size {
        let mut sum = 0.0;
        for h_idx in 0..config.hidden_size {
            sum += manual_embedding[h_idx] * lm_head_weights[h_idx * config.vocab_size + v_idx];
        }
        manual_logits[v_idx] = sum;
    }
    
    // Compare with model's final logits
    let model_logits = model.forward(&[tokens[0]], 0)?;
    let model_logits = &model_logits[..config.vocab_size];
    
    let mut max_diff = 0.0;
    for i in 0..config.vocab_size {
        let diff = (manual_logits[i] - model_logits[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    println!("   Manual LM head vs model LM head max diff: {:.6}", max_diff);
    if max_diff > 1e-5 {
        println!("   ⚠️  WARNING: Large difference suggests LM head orientation issue!");
    } else {
        println!("   ✅ LM head orientation appears correct");
    }

    // Test 5: KV Cache performance
    println!("\n--- Test 5: KV Cache Performance ---");
    
    // Clear KV cache
    model.clear_kv_cache();
    
    // First token (should populate KV cache)
    let start = Instant::now();
    let _logits1 = model.forward(&[tokens[0]], 0)?;
    let duration1 = start.elapsed();
    println!("   First token (KV cache populate): {:?}", duration1);
    
    // Second token (should use KV cache)
    let start = Instant::now();
    let _logits2 = model.forward(&[tokens[0]], 1)?;
    let duration2 = start.elapsed();
    println!("   Second token (KV cache use): {:?}", duration2);
    
    if duration2 > duration1 {
        println!("   ⚠️  WARNING: Second token slower than first - KV cache not working!");
    } else {
        println!("   ✅ KV cache appears to be working");
    }

    // Test 6: Check for memory allocations
    println!("\n--- Test 6: Memory Allocation Check ---");
    
    // Run multiple forward passes and check for memory growth
    let start = Instant::now();
    for i in 0..10 {
        let _logits = model.forward(&[tokens[0]], i)?;
    }
    let duration = start.elapsed();
    println!("   10 forward passes took: {:?}", duration);
    println!("   Average per pass: {:?}", duration / 10);

    Ok(())
}
