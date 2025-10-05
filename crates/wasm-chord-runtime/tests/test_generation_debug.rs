/// Debug text generation to see what's happening
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn get_model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut path = PathBuf::from(manifest_dir);
    path.pop();
    path.pop();

    let q8_path = path.join("models/tinyllama-q8.gguf");
    if q8_path.exists() {
        return q8_path;
    }

    let q4km_path = path.join("models/tinyllama-q4km.gguf");
    if q4km_path.exists() {
        q4km_path
    } else {
        path.join("models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
    }
}

#[test]
#[ignore]
fn debug_generation() {
    let model_path = get_model_path();
    println!("🔍 Debug generation test");

    // Load model
    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse header");

    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    let tokenizer = Tokenizer::from_gguf(&meta).expect("Failed to create tokenizer");
    println!(
        "✅ Tokenizer: {} tokens, {} merges",
        tokenizer.vocab_size(),
        meta.metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0)
    );

    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset().expect("Failed to get data offset");
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let file = File::open(&model_path).expect("Failed to reopen");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Re-parse");

    model.load_from_gguf(&mut tensor_loader, &mut parser).expect("Failed to load weights");
    println!("✅ Weights loaded\n");

    // Test encoding
    let prompt = "Hello";
    println!("📝 Prompt: {:?}", prompt);
    let tokens = tokenizer.encode(prompt, true).expect("Failed to encode");
    println!("   Encoded to {} tokens: {:?}", tokens.len(), tokens);

    for &token_id in &tokens {
        if let Some(token_str) = tokenizer.id_to_token(token_id) {
            println!("     [{}] = {:?}", token_id, token_str);
        }
    }

    // Manual generation loop with debug output
    println!("\n🔄 Generating tokens:");
    model.clear_kv_cache();

    let mut all_tokens = tokens.clone();
    let position = 0;

    // Process prompt
    let _logits = model.forward(&all_tokens, position).expect("Forward failed");

    for step in 0..10 {
        let last_token = *all_tokens.last().unwrap();
        let current_pos = all_tokens.len() - 1;

        // Forward pass
        let logits = model.forward(&[last_token], current_pos).expect("Forward failed");
        let last_logits = &logits[(logits.len() - model.config.vocab_size)..];

        // Get top 5 logits
        let mut indexed_logits: Vec<(usize, f32)> =
            last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n  Step {}: Top 5 candidates:", step);
        for (idx, &(token_id, logit_val)) in indexed_logits.iter().take(5).enumerate() {
            let token_str = tokenizer.id_to_token(token_id as u32).unwrap_or("<???>");
            println!("    {}. [{}] {:?} (logit: {:.4})", idx + 1, token_id, token_str, logit_val);
        }

        // Sample (greedy)
        let next_token = model.sample(last_logits, 0.0, 1.0, 0).expect("Sampling failed");
        let next_token_str = tokenizer.id_to_token(next_token).unwrap_or("<???>");
        println!("  → Selected: [{}] {:?}", next_token, next_token_str);

        if next_token == tokenizer.special_tokens().eos_token_id {
            println!("  (EOS reached)");
            break;
        }

        all_tokens.push(next_token);
    }

    // Decode final result
    let decoded = tokenizer.decode(&all_tokens, true).expect("Decode failed");
    println!("\n📝 Final result: {:?}", decoded);
}
