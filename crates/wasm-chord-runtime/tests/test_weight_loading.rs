/// Test loading real weights from TinyLlama GGUF
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use wasm_chord_core::{GGUFParser, TensorLoader};
use wasm_chord_runtime::{Model, TransformerConfig};

fn get_model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to find workspace root")
        .join("models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
}

#[test]
#[ignore]
fn test_load_real_weights() {
    let model_path = get_model_path();
    println!("📂 Loading weights from: {}", model_path.display());

    // Parse GGUF
    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);

    let meta = parser.parse_header().expect("Failed to parse header");
    println!("✅ Parsed GGUF: {} tensors", meta.tensor_count);

    // Extract config
    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    println!("📋 Config: {} layers, {} vocab", config.num_layers, config.vocab_size);

    // Create model
    let mut model = Model::new(config.clone());
    println!("✅ Model created");

    // Create tensor loader and register tensors
    let data_offset = parser.tensor_data_offset().expect("Failed to get data offset");
    let mut tensor_loader = TensorLoader::new(data_offset);

    println!("📦 Registering {} tensors...", meta.tensors.len());
    for (i, tensor_desc) in meta.tensors.iter().enumerate() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );

        if i < 5 || tensor_desc.name.contains("token_embd") {
            println!(
                "   {}. {} - {:?} ({:?}, {} bytes)",
                i + 1,
                tensor_desc.name,
                tensor_desc.shape,
                tensor_desc.dtype,
                tensor_desc.size_bytes
            );
        }
    }

    // Reopen file for tensor loading (parser consumed reader)
    let file = File::open(&model_path).expect("Failed to reopen model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header().expect("Re-parse header");

    println!("🔄 Loading weights into model...");
    let load_result = model.load_from_gguf(&mut tensor_loader, &mut parser);

    match load_result {
        Ok(_) => {
            println!("✅ Weights loaded successfully!");

            // Test that embeddings are loaded (should not be all zeros)
            let embedding_sum: f32 = model.token_embeddings.iter().take(100).sum();
            println!("   Embedding sample sum (first 100): {}", embedding_sum);
            assert_ne!(embedding_sum, 0.0, "Embeddings should not be all zeros");
        }
        Err(e) => {
            println!("❌ Failed to load weights: {:?}", e);
            panic!("Weight loading failed");
        }
    }
}

#[test]
#[ignore]
fn test_inference_with_real_weights() {
    let model_path = get_model_path();
    println!("🚀 Testing inference with real weights");

    // Load model with weights
    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);

    let meta = parser.parse_header().expect("Failed to parse header");
    let config_data = parser.extract_config().expect("Failed to extract config");
    let config: TransformerConfig = config_data.into();

    let mut model = Model::new(config.clone());

    // Load weights
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

    println!("✅ Weights loaded");

    // Run inference
    let test_tokens = vec![1u32]; // BOS token
    println!("🔄 Running inference with token: {:?}", test_tokens);

    let start = std::time::Instant::now();
    let logits = model.forward(&test_tokens, 0).expect("Forward pass failed");
    let duration = start.elapsed();

    println!("✅ Forward pass complete in {:?}", duration);
    println!("   Output: {} logits", logits.len());

    // Get logits for last token
    let last_token_logits = &logits[(logits.len() - model.config.vocab_size)..];

    // Debug: Check for inf/nan
    let num_inf = last_token_logits.iter().filter(|&&x| x.is_infinite()).count();
    let num_nan = last_token_logits.iter().filter(|&&x| x.is_nan()).count();
    println!("   Debug: {} inf, {} nan values", num_inf, num_nan);

    // Check embedding values
    let emb_sum: f32 = model.token_embeddings.iter().take(100).sum();
    let emb_has_inf = model.token_embeddings.iter().take(1000).any(|&x| x.is_infinite());
    let emb_has_nan = model.token_embeddings.iter().take(1000).any(|&x| x.is_nan());
    println!(
        "   Embedding sum (first 100): {}, has_inf: {}, has_nan: {}",
        emb_sum, emb_has_inf, emb_has_nan
    );
    println!("   First 10 embeddings: {:?}", &model.token_embeddings[0..10]);

    // Sample next token
    let next_token = model.sample(last_token_logits, 0.0, 1.0, 0).expect("Sampling failed");

    println!("🎲 Next token (greedy): {}", next_token);
    println!(
        "   Logits range: [{:.4}, {:.4}]",
        last_token_logits.iter().copied().fold(f32::INFINITY, f32::min),
        last_token_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    assert!(next_token < model.config.vocab_size as u32);
}
