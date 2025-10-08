/// Logit Analysis Tool
/// This tool analyzes the logit values for specific tokens to understand the prediction
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Logit Analysis Tool");
    println!("=====================\n");

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

    // Test with "Hello"
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false)?;
    let token_id = tokens[0];

    println!("\n🔍 Testing with prompt: '{}'", prompt);
    println!("   Token ID: {}", token_id);
    println!("   Token text: {:?}", tokenizer.id_to_token(token_id));

    // Run forward pass
    let logits = model.forward(&tokens, 0)?;

    // Analyze specific tokens
    let target_tokens = vec![
        ("▁concaten", 16125),
        ("▁Yes", 3869),
        ("▁Hello", 15043),
        ("▁Мос", 9439),
        ("Dict", 21533),
    ];

    println!("\n📊 Logit Analysis:");
    println!("==================");

    for (token_name, token_id) in target_tokens {
        if token_id < logits.len() as u32 {
            let logit = logits[token_id as usize];
            println!("{:12} (ID: {:5}) → logit: {:.6}", token_name, token_id, logit);
        }
    }

    // Find top 10 tokens
    let mut token_logits: Vec<(u32, f32)> =
        logits.iter().enumerate().map(|(i, &logit)| (i as u32, logit)).collect();
    token_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 10 Predictions:");
    println!("=====================");

    for (i, (token_id, logit)) in token_logits.iter().take(10).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id).unwrap_or("<unknown>");
        println!("{:2}. {:12} (ID: {:5}) → logit: {:.6}", i + 1, token_text, token_id, logit);
    }

    // Check if our prediction matches the expected
    let predicted_token_id = token_logits[0].0;
    let predicted_token_text = tokenizer.id_to_token(predicted_token_id).unwrap_or("<unknown>");
    let expected_token_id = 3869; // "▁Yes"
    let expected_logit = logits[expected_token_id as usize];

    println!("\n🎯 Prediction Analysis:");
    println!("=======================");
    println!(
        "Our prediction: '{}' (ID: {}) → logit: {:.6}",
        predicted_token_text, predicted_token_id, token_logits[0].1
    );
    println!(
        "Expected (Ollama): '{}' (ID: {}) → logit: {:.6}",
        "▁Yes", expected_token_id, expected_logit
    );

    let logit_diff = token_logits[0].1 - expected_logit;
    println!("Logit difference: {:.6}", logit_diff);

    if predicted_token_id == expected_token_id {
        println!("✅ PREDICTION MATCHES!");
    } else {
        println!("❌ PREDICTION MISMATCH!");
        println!("   Our model predicts '{}' instead of '{}'", predicted_token_text, "▁Yes");
    }

    Ok(())
}
