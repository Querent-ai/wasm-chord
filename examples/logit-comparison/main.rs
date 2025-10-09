/// Numerical logit comparison tool
///
/// This tool will compare our model's logits with Ollama's logits
/// to identify exactly where the divergence occurs.
use std::fs::File;
use std::io::BufReader;
use std::process::Command;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Numerical Logit Comparison Tool");
    println!("==================================\n");

    // Load our model
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    println!("📂 Loading model: {}", model_path);

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded with {} tokens", tokenizer.vocab_size());

    // Load weights
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

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model loaded\n");

    // Test with simple prompt
    let prompt = "Hello";
    println!("🎯 Testing prompt: \"{}\"", prompt);

    // Get our logits
    let tokens = tokenizer.encode(prompt, false)?;
    println!("📝 Our tokens: {:?}", tokens);

    // Forward pass to get logits
    let logits = model.forward(&tokens, 0)?;
    println!("📊 Our logits shape: {}", logits.len());

    // Get top 10 logits
    let mut indexed_logits: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Our top 10 logits:");
    for (i, (token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id as u32);
        println!("  {}: token {} = {:.6} ({:?})", i + 1, token_id, logit, token_text);
    }

    // Now get Ollama's logits using llama.cpp directly
    println!("\n🔄 Getting Ollama's logits...");

    // Create a temporary file with our prompt
    std::fs::write("/tmp/test_prompt.txt", prompt)?;

    // Run llama.cpp to get logits
    let output = Command::new("ollama")
        .args(&["run", "tinyllama", "--verbose"])
        .arg(format!("--prompt {}", prompt))
        .arg("--num-predict 1")
        .output()?;

    if !output.status.success() {
        println!("❌ Ollama failed: {}", String::from_utf8_lossy(&output.stderr));
        return Ok(());
    }

    println!("✅ Ollama output: {}", String::from_utf8_lossy(&output.stdout));

    // Alternative: Use llama.cpp directly if available
    println!("\n🔍 Trying llama.cpp directly...");

    // Check if llama.cpp is available
    let llama_output = Command::new("which").arg("llama-cpp").output();

    if let Ok(output) = llama_output {
        if output.status.success() {
            println!("✅ Found llama-cpp, running comparison...");

            // Run llama.cpp with logit output
            let llama_result = Command::new("llama-cpp")
                .args(&["-m", model_path, "-p", prompt, "--logits", "-n", "1"])
                .output();

            if let Ok(result) = llama_result {
                if result.status.success() {
                    println!("📊 Llama.cpp output: {}", String::from_utf8_lossy(&result.stdout));
                } else {
                    println!("❌ Llama.cpp failed: {}", String::from_utf8_lossy(&result.stderr));
                }
            }
        } else {
            println!("⚠️  llama-cpp not found, skipping direct comparison");
        }
    }

    // Manual comparison with known good values
    println!("\n📋 Manual Analysis:");
    println!("Our top token: {} ({:.6})", indexed_logits[0].0, indexed_logits[0].1);
    println!("Expected for 'Hello': Should be a continuation like 'there', 'world', 'how', etc.");
    println!("Our result: {:?}", tokenizer.id_to_token(indexed_logits[0].0 as u32));

    // Check if our top tokens make sense
    println!("\n🔍 Token Analysis:");
    for (i, (token_id, _)) in indexed_logits.iter().take(5).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id as u32);
        println!(
            "  Top {}: {:?} - {}",
            i + 1,
            token_text,
            if token_text
                .as_ref()
                .map(|s| s.contains("Hello") || s.contains("there") || s.contains("world"))
                .unwrap_or(false)
            {
                "✅ Makes sense"
            } else {
                "❌ Gibberish"
            }
        );
    }

    Ok(())
}
