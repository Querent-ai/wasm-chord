/// Simple first token comparison
///
/// Compare our first token prediction with Ollama's first token prediction
use std::fs::File;
use std::io::BufReader;
use std::process::Command;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 First Token Comparison");
    println!("========================\n");

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

    // Get top 5 logits
    let mut indexed_logits: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Our top 5 predictions after \"Hello\":");
    for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id as u32);
        println!("  {}: token {} = {:.6} ({:?})", i + 1, token_id, logit, token_text);
    }

    // Get Ollama's first token prediction
    println!("\n🔄 Getting Ollama's first token prediction...");

    // Use a simple approach: run Ollama with very short output
    let output =
        Command::new("ollama").args(&["run", "tinyllama", "-n", "1", "-p", prompt]).output()?;

    if !output.status.success() {
        println!("❌ Ollama failed: {}", String::from_utf8_lossy(&output.stderr));
        return Ok(());
    }

    let ollama_output = String::from_utf8_lossy(&output.stdout).trim().to_string();
    println!("✅ Ollama's first token: \"{}\"", ollama_output);

    // Tokenize Ollama's output to see what token it predicted
    let ollama_tokens = tokenizer.encode(&ollama_output, false)?;
    println!("📝 Ollama's tokens: {:?}", ollama_tokens);

    if let Some(&first_token) = ollama_tokens.first() {
        let token_text = tokenizer.id_to_token(first_token);
        println!("🔍 Ollama's first token: {} ({:?})", first_token, token_text);

        // Check if this token is in our top predictions
        let mut found_in_top = false;
        for (i, (token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
            if *token_id == first_token as usize {
                println!(
                    "✅ Ollama's token {} found in our top {} at position {}",
                    first_token,
                    10,
                    i + 1
                );
                found_in_top = true;
                break;
            }
        }

        if !found_in_top {
            println!("❌ Ollama's token {} NOT found in our top 10 predictions!", first_token);
            println!("   This indicates a fundamental bug in our implementation.");
        }
    }

    // Analysis
    println!("\n📋 Analysis:");
    println!(
        "Our top prediction: {:?} (token {})",
        tokenizer.id_to_token(indexed_logits[0].0 as u32),
        indexed_logits[0].0
    );
    println!("Ollama's prediction: \"{}\"", ollama_output);

    // Check if our predictions make sense
    println!("\n🔍 Sense Check:");
    for (i, (token_id, _)) in indexed_logits.iter().take(3).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id as u32);
        let makes_sense = token_text
            .as_ref()
            .map(|s| {
                s.contains("there")
                    || s.contains("world")
                    || s.contains("how")
                    || s.contains("are")
                    || s.contains("you")
                    || s.contains("my")
                    || s.contains("name")
                    || s.contains("is")
                    || s.contains("I")
            })
            .unwrap_or(false);

        println!(
            "  Top {}: {:?} - {}",
            i + 1,
            token_text,
            if makes_sense { "✅ Makes sense" } else { "❌ Gibberish" }
        );
    }

    Ok(())
}
