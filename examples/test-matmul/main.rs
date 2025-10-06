/// Test matmul with loaded GGUF weights
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader};
use wasm_chord_cpu::matmul_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Matmul with GGUF Weights\n");

    let model_path = "models/tinyllama-q8.gguf";
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    // Register all tensors
    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Reopen for loading
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    // Load token_embd and output weights
    let token_embd = tensor_loader.load_tensor("token_embd.weight", &mut parser)?.to_vec();
    let output_weight = tensor_loader.load_tensor("output.weight", &mut parser)?.to_vec();

    println!("✅ Loaded token_embd: {} elements", token_embd.len());
    println!("✅ Loaded output.weight: {} elements\n", output_weight.len());

    // Test: Get embedding for token 1 (BOS)
    let hidden_size = 2048;
    let vocab_size = 32000;
    let token_id = 1_usize;

    let emb_start = token_id * hidden_size;
    let emb_end = emb_start + hidden_size;
    let embedding = &token_embd[emb_start..emb_end];

    println!("Token {} embedding (first 10): {:?}", token_id, &embedding[..10]);
    println!(
        "Embedding stats: mean={:.6}, min={:.6}, max={:.6}\n",
        embedding.iter().sum::<f32>() / hidden_size as f32,
        embedding.iter().copied().fold(f32::INFINITY, f32::min),
        embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Test matmul: embedding @ output_weight
    // embedding: [1, 2048]
    // output_weight: [2048, 32000]
    // result: [1, 32000]
    let mut logits = vec![0.0f32; vocab_size];

    println!("Performing matmul: [1, 2048] @ [2048, 32000] = [1, 32000]");
    matmul_f32(embedding, output_weight, &mut logits, 1, hidden_size, vocab_size)?;

    println!("\n✅ Matmul complete!");
    println!("Logits (first 10): {:?}", &logits[..10]);
    println!(
        "Logits stats: mean={:.6}, min={:.6}, max={:.6}",
        logits.iter().sum::<f32>() / vocab_size as f32,
        logits.iter().copied().fold(f32::INFINITY, f32::min),
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Find top 5 logits
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 tokens:");
    for (i, (token_id, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}. Token {}: logit={:.4}", i + 1, token_id, logit);
    }

    Ok(())
}
