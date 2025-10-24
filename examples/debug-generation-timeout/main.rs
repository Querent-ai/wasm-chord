/// Debug generation timeout issue
///
/// Minimal test with timing diagnostics

use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug Generation Timeout\n");

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf".to_string());
    
    println!("📂 Model: {}", model_path);

    // === Parse Header ===
    let start = Instant::now();
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    println!("✅ Header parsed: {:?}", start.elapsed());

    // === Extract Config ===
    let start = Instant::now();
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    println!("✅ Config extracted: {:?} ({} layers)", start.elapsed(), config.num_layers);

    // === Create Tokenizer ===
    let start = Instant::now();
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer created: {:?} ({} tokens)", start.elapsed(), tokenizer.vocab_size());

    // === Setup Tensor Loader ===
    let start = Instant::now();
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);
    
    println!("   Registering {} tensors...", meta.tensors.len());
    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }
    println!("✅ Tensors registered: {:?}", start.elapsed());

    // === Create Model ===
    let start = Instant::now();
    let mut model = Model::new(config.clone());
    println!("✅ Model created: {:?}", start.elapsed());

    // === Load Weights (THIS IS WHERE IT HANGS) ===
    println!("\n🔥 Starting weight loading...");
    let start = Instant::now();
    
    // Reopen file
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;
    
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Weights loaded: {:?}\n", start.elapsed());

    // === Test Single Forward Pass ===
    println!("🧪 Testing single forward pass...");
    let start = Instant::now();
    
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, true)?;
    println!("   Prompt: \"{}\" → {} tokens", prompt, tokens.len());
    
    let logits = model.forward(&tokens)?;
    println!("✅ Forward pass: {:?} (output shape: {})", start.elapsed(), logits.len());

    // === Test Single Token Generation ===
    println!("\n🧪 Testing single token generation...");
    let start = Instant::now();
    
    let next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    let decoded = tokenizer.decode(&[next_token as u32], true)?;
    println!("✅ Token generated: {:?}", start.elapsed());
    println!("   Next token: {} → \"{}\"", next_token, decoded);

    println!("\n🎉 All tests passed! Generation is working.");
    
    Ok(())
}

