/// Check the actual GGUF embedding format
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Check Actual GGUF Embedding Format");
    println!("=====================================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b.Q4_0.gguf";
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

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

    // Test different embedding formats
    let token_id = 1u32;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;

    println!("\n🧪 Testing different embedding formats for token {}:", token_id);
    println!("  hidden_size = {}, vocab_size = {}", hidden_size, vocab_size);

    // Format 1: [vocab_size, hidden_size] - each token is a contiguous block
    println!("\n📊 Format 1: [vocab_size, hidden_size] (contiguous blocks)");
    let start1 = (token_id as usize) * hidden_size;
    let end1 = start1 + hidden_size;
    if end1 <= model.token_embeddings.len() {
        let emb1: Vec<f32> = model.token_embeddings[start1..end1].to_vec();
        println!("  Indices: {} to {}", start1, end1);
        println!("  First 10: {:?}", &emb1[..10.min(emb1.len())]);
        let sum: f32 = emb1.iter().sum();
        println!("  Sum: {:.6}, Mean: {:.6}", sum, sum / hidden_size as f32);
    }

    // Format 2: [hidden_size, vocab_size] - each token is scattered across rows
    println!("\n📊 Format 2: [hidden_size, vocab_size] (scattered across rows)");
    let mut emb2 = vec![0.0; hidden_size];
    for j in 0..hidden_size {
        let idx = j * vocab_size + (token_id as usize);
        if idx < model.token_embeddings.len() {
            emb2[j] = model.token_embeddings[idx];
        }
    }
    println!("  First 10: {:?}", &emb2[..10.min(emb2.len())]);
    let sum: f32 = emb2.iter().sum();
    println!("  Sum: {:.6}, Mean: {:.6}", sum, sum / hidden_size as f32);

    // Format 3: Check if it's actually [vocab_size, hidden_size] but transposed
    println!("\n📊 Format 3: Check if embeddings are already transposed");
    println!("  Raw first 20 values: {:?}", &model.token_embeddings[..20]);

    // Test token 0 vs token 1 to see the pattern
    println!("\n📊 Compare token 0 vs token 1:");
    let token0_start = 0 * hidden_size;
    let token0_end = token0_start + hidden_size;
    let token1_start = 1 * hidden_size;
    let token1_end = token1_start + hidden_size;

    if token1_end <= model.token_embeddings.len() {
        let token0_emb: Vec<f32> = model.token_embeddings[token0_start..token0_end].to_vec();
        let token1_emb: Vec<f32> = model.token_embeddings[token1_start..token1_end].to_vec();

        println!("  Token 0 (indices {} to {}): {:?}", token0_start, token0_end, &token0_emb[..5]);
        println!("  Token 1 (indices {} to {}): {:?}", token1_start, token1_end, &token1_emb[..5]);

        // Check if they're different (they should be for different tokens)
        let diff: f32 = token0_emb.iter().zip(token1_emb.iter()).map(|(a, b)| (a - b).abs()).sum();
        println!("  Difference between token 0 and 1: {:.6}", diff);

        if diff < 1e-6 {
            println!("  ❌ Tokens 0 and 1 are identical - this suggests wrong format!");
        } else {
            println!("  ✅ Tokens 0 and 1 are different - format might be correct");
        }
    }

    Ok(())
}
