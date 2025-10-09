/// Comprehensive forward pass debugging tool
/// This tracks intermediate values to identify where the model goes wrong
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn print_stats(name: &str, values: &[f32]) {
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let std_dev =
        (values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt();

    println!("  {}: min={:.6}, max={:.6}, mean={:.6}, std={:.6}", name, min, max, mean, std_dev);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Forward Pass Deep Debug");
    println!("{}", "=".repeat(60));

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf".to_string());

    println!("\n📂 Loading model: {}", model_path);

    // Load model
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    println!("✅ Config loaded:");
    println!(
        "   vocab_size={}, hidden_size={}, num_layers={}",
        config.vocab_size, config.hidden_size, config.num_layers
    );

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

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

    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model weights loaded\n");

    // Test prompt
    let prompt = "Hello";
    println!("📝 Testing prompt: {:?}", prompt);

    let tokens = tokenizer.encode(prompt, true)?; // With BOS token
    println!("✅ Tokens: {:?}", tokens);
    for &token_id in &tokens {
        let token_text = tokenizer.id_to_token(token_id).unwrap_or("<unknown>");
        println!("   {} -> {:?}", token_id, token_text);
    }

    println!("\n🔄 Running forward pass with detailed tracking...\n");

    // Run forward pass
    let logits = model.forward(&tokens, 0)?;

    println!("\n✅ Forward pass completed");
    print_stats("Final logits", &logits);

    // Check logit range
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().copied().fold(f32::INFINITY, f32::min);
    println!("\n📊 Logit range: [{:.6}, {:.6}]", min_logit, max_logit);

    if max_logit > 100.0 || min_logit < -100.0 {
        println!("⚠️  WARNING: Extreme logit values detected!");
    }

    // Count out-of-vocab predictions
    let mut oov_count = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if i >= config.vocab_size as usize {
            oov_count += 1;
        }
    }
    println!("📊 Logits array size: {} (should be vocab_size={})", logits.len(), config.vocab_size);
    if logits.len() > config.vocab_size as usize {
        println!("⚠️  WARNING: Logits array is larger than vocab size!");
    }

    // Find top tokens
    let mut indexed_logits: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 20 predictions:");
    for (rank, (token_id, logit)) in indexed_logits.iter().take(20).enumerate() {
        if *token_id < config.vocab_size as usize {
            let token_text = tokenizer.id_to_token(*token_id as u32).unwrap_or("<unknown>");
            println!("   {:2}. {} (id={}, logit={:.6})", rank + 1, token_text, token_id, logit);
        } else {
            println!(
                "   {:2}. <OUT-OF-VOCAB:{}> (id={}, logit={:.6})",
                rank + 1,
                token_id,
                token_id,
                logit
            );
        }
    }

    // Check expected conversational tokens
    println!("\n🔍 Checking expected conversational tokens:");
    let expected_tokens = vec![
        ("!", vec!["!", "▁!"]),
        ("?", vec!["?", "▁?"]),
        (" there", vec!["▁there"]),
        (" world", vec!["▁world"]),
        (",", vec![",", "▁,"]),
        (".", vec![".", "▁."]),
        (" I", vec!["▁I"]),
        (" How", vec!["▁How"]),
        (" Nice", vec!["▁Nice"]),
        (" Hi", vec!["▁Hi"]),
    ];

    for (desc, variants) in &expected_tokens {
        for variant in variants {
            if let Some(token_id) = tokenizer.token_to_id(variant) {
                let logit = logits[token_id as usize];
                let rank = indexed_logits
                    .iter()
                    .position(|(id, _)| *id == token_id as usize)
                    .unwrap_or(usize::MAX);
                println!("   {:10} ({}): logit={:8.4}, rank={}", desc, variant, logit, rank + 1);
            }
        }
    }

    // Check token embeddings
    println!("\n🔍 Checking token embedding layer:");
    println!("   (This requires access to model internals - showing what we can)");

    Ok(())
}
