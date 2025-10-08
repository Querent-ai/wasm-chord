/// Direct comparison with Ollama TinyLlama
/// This loads the exact same model and compares first token generation

use std::fs::File;
use std::io::{BufReader, Seek};
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Direct Comparison with Ollama TinyLlama");
    println!("==========================================\n");

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";
    
    // Test 1: Load model and check config
    println!("1️⃣ Loading Model Configuration");
    println!("-------------------------------");
    
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    
    println!("✅ GGUF file loaded successfully");
    println!("   Architecture: {:?}", meta.architecture);
    println!("   Tensor count: {}", meta.tensor_count);
    println!("   Vocab size: {:?}", meta.vocab_size);
    
    // Extract config from GGUF
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    
    println!("✅ Model configuration:");
    println!("   Vocab size: {}", config.vocab_size);
    println!("   Hidden size: {}", config.hidden_size);
    println!("   Num layers: {}", config.num_layers);
    println!("   Num heads: {}", config.num_heads);
    println!("   Num KV heads: {}", config.num_kv_heads);
    println!("   Intermediate size: {}", config.intermediate_size);
    println!("   RMS norm eps: {}", config.rms_norm_eps);
    println!("   RoPE theta: {}", config.rope_theta);
    
    // Test 2: Load tokenizer
    println!("\n2️⃣ Loading Tokenizer");
    println!("--------------------");
    
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());
    
    // Test 3: Tokenize "Hello"
    println!("\n3️⃣ Tokenizing 'Hello'");
    println!("---------------------");
    
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false)?;
    println!("✅ Tokens for '{}': {:?}", prompt, tokens);
    
    // Test 4: Load model weights
    println!("\n4️⃣ Loading Model Weights");
    println!("-------------------------");
    
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);
    
    // Register all tensors
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
    
    // Load weights
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model weights loaded successfully");
    
    // Test 5: Generate first token
    println!("\n5️⃣ Generating First Token");
    println!("-------------------------");
    
    let logits = model.forward(&tokens, 0)?;
    println!("✅ Forward pass completed");
    println!("   Logits shape: {}", logits.len());
    
    // Find top token
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_token = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > max_logit {
            max_logit = logit;
            max_token = i;
        }
    }
    
    let token_text = tokenizer.id_to_token(max_token as u32);
    let token_str = token_text.unwrap_or("<unknown>");
    println!("✅ Top token: {} (id: {}, logit: {:.6})", token_str, max_token, max_logit);
    
    // Compare with Ollama
    println!("\n6️⃣ Comparison with Ollama");
    println!("--------------------------");
    println!("Ollama output: 'Yes'");
    println!("Our output: '{}'", token_str);
    
    if token_text == Some("Yes") {
        println!("✅ MATCH! Our implementation produces the same first token as Ollama");
    } else {
        println!("❌ MISMATCH! Our implementation differs from Ollama");
        println!("   This indicates a bug in our implementation");
    }
    
    Ok(())
}