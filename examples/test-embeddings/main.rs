/// Simple test to verify token embedding lookup
///
/// This will test the BOS token (ID=1) embedding lookup before and after transpose
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Token Embedding Lookup Test");
    println!("==============================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b.Q4_0.gguf";
    println!("📂 Loading model: {}", model_path);

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    let _tokenizer = Tokenizer::from_gguf(&meta)?;

    println!(
        "✅ Config loaded: {} layers, {} vocab, {} hidden_size",
        config.num_layers, config.vocab_size, config.hidden_size
    );

    // Load model
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let mut model = Model::new(config.clone());
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model loaded successfully");

    // Test BOS token (ID=1) embedding lookup
    let bos_token_id = 1u32;
    println!("\n🧪 Testing BOS token (ID={}) embedding lookup:", bos_token_id);

    // Test the current embedding lookup method
    let hidden_size = config.hidden_size;
    let mut embedding = vec![0.0; hidden_size];

    // Current method: assumes [vocab_size, hidden_size] storage
    let emb_start = (bos_token_id as usize) * hidden_size;
    let emb_end = emb_start + hidden_size;

    if emb_end <= model.token_embeddings.len() {
        embedding.copy_from_slice(&model.token_embeddings[emb_start..emb_end]);
        println!("  Current method (assumes [vocab_size, hidden_size]):");
        println!("    Indices: {} to {}", emb_start, emb_end);
        println!("    First 10 values: {:?}", &embedding[..10.min(embedding.len())]);
        println!(
            "    Sum: {:.6}, Mean: {:.6}",
            embedding.iter().sum::<f32>(),
            embedding.iter().sum::<f32>() / embedding.len() as f32
        );
    } else {
        println!("  ❌ Current method: Token {} out of bounds", bos_token_id);
    }

    // Test correct method: assumes [hidden_size, vocab_size] storage
    let mut embedding_correct = vec![0.0; hidden_size];
    for j in 0..hidden_size {
        let emb_idx = j * config.vocab_size + (bos_token_id as usize);
        if emb_idx < model.token_embeddings.len() {
            embedding_correct[j] = model.token_embeddings[emb_idx];
        }
    }

    println!("  Correct method (assumes [hidden_size, vocab_size]):");
    println!("    First 10 values: {:?}", &embedding_correct[..10.min(embedding_correct.len())]);
    println!(
        "    Sum: {:.6}, Mean: {:.6}",
        embedding_correct.iter().sum::<f32>(),
        embedding_correct.iter().sum::<f32>() / embedding_correct.len() as f32
    );

    // Compare the two methods
    let diff: f32 =
        embedding.iter().zip(embedding_correct.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("  Difference between methods: {:.6}", diff);

    if diff < 1e-6 {
        println!("  ✅ Both methods produce the same result - current method is correct");
    } else {
        println!("  ❌ Methods differ - current method is WRONG!");
        println!("  🎯 This confirms the embedding lookup bug!");
    }

    // Test with a few more tokens
    println!("\n🧪 Testing other tokens:");
    for token_id in [0, 2, 10, 100] {
        let mut emb_current = vec![0.0; hidden_size];
        let mut emb_correct = vec![0.0; hidden_size];

        // Current method
        let start = (token_id as usize) * hidden_size;
        let end = start + hidden_size;
        if end <= model.token_embeddings.len() {
            emb_current.copy_from_slice(&model.token_embeddings[start..end]);
        }

        // Correct method
        for j in 0..hidden_size {
            let idx = j * config.vocab_size + (token_id as usize);
            if idx < model.token_embeddings.len() {
                emb_correct[j] = model.token_embeddings[idx];
            }
        }

        let diff: f32 =
            emb_current.iter().zip(emb_correct.iter()).map(|(a, b)| (a - b).abs()).sum();

        println!(
            "  Token {}: diff = {:.6} {}",
            token_id,
            diff,
            if diff < 1e-6 { "✅" } else { "❌" }
        );
    }

    Ok(())
}
