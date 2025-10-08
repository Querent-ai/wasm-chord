use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Raw Argmax Test (No Sampling)");
    println!("===============================\n");

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

    // Test different prompts
    let test_prompts = vec!["Hello", "Hi", "The", "cat", "dog"];

    for prompt in test_prompts {
        println!("🎯 Testing prompt: \"{}\"", prompt);

        // Encode prompt
        let tokens = tokenizer.encode(prompt, false)?;
        println!("  Tokens: {:?}", tokens);

        // Get logits for last token
        match model.forward(&tokens, tokens.len() - 1) {
            Ok(logits) => {
                // Find argmax
                let mut max_logit = f32::NEG_INFINITY;
                let mut argmax_token = 0;

                for (i, &logit) in logits.iter().enumerate() {
                    if logit > max_logit {
                        max_logit = logit;
                        argmax_token = i;
                    }
                }

                println!("  Argmax token: {} (logit: {:.3})", argmax_token, max_logit);

                // Decode the argmax token
                match tokenizer.decode(&[argmax_token as u32], true) {
                    Ok(txt) => println!("  Decoded: {:?}", txt),
                    Err(_) => println!("  Decode failed"),
                }

                // Show top 5 tokens
                let mut indexed_logits: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                println!("  Top 5 tokens:");
                for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
                    match tokenizer.decode(&[*token_id as u32], true) {
                        Ok(txt) => {
                            println!("    {}: {} = {:.3} ({:?})", i + 1, token_id, logit, txt)
                        }
                        Err(_) => {
                            println!("    {}: {} = {:.3} (<decode-fail>)", i + 1, token_id, logit)
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Forward pass failed: {}", e);
            }
        }
        println!();
    }

    Ok(())
}
