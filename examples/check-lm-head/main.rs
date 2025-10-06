/// Check if lm_head has bias toward token 19762
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Checking lm_head weights for bias\n");

    let model_path = "models/tinyllama-q8.gguf";
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

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

    let output_weight = tensor_loader.load_tensor("output.weight", &mut parser)?;

    let hidden_size = 2048;
    let vocab_size = 32000;

    println!("✅ Loaded output.weight: {} elements", output_weight.len());
    println!("Expected: {} x {} = {}\n", hidden_size, vocab_size, hidden_size * vocab_size);

    // Check column for token 19762
    let token_19762_col = 19762;
    println!("Analyzing column for token {}:", token_19762_col);

    let mut col_values: Vec<f32> = Vec::with_capacity(hidden_size);
    for row in 0..hidden_size {
        let idx = row * vocab_size + token_19762_col;
        col_values.push(output_weight[idx]);
    }

    let mean: f32 = col_values.iter().sum::<f32>() / hidden_size as f32;
    let min = col_values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = col_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let abs_mean: f32 = col_values.iter().map(|&x| x.abs()).sum::<f32>() / hidden_size as f32;

    println!("  Mean: {:.6}", mean);
    println!("  Abs mean: {:.6}", abs_mean);
    println!("  Range: [{:.6}, {:.6}]", min, max);
    println!("  First 10: {:?}\n", &col_values[..10]);

    // Compare with a few other random tokens
    for &token_id in &[100, 500, 1000, 5000, 10000, 20000] {
        let mut col: Vec<f32> = Vec::with_capacity(hidden_size);
        for row in 0..hidden_size {
            let idx = row * vocab_size + token_id;
            col.push(output_weight[idx]);
        }
        let mean: f32 = col.iter().sum::<f32>() / hidden_size as f32;
        let abs_mean: f32 = col.iter().map(|&x| x.abs()).sum::<f32>() / hidden_size as f32;
        println!("Token {}: mean={:.6}, abs_mean={:.6}", token_id, mean, abs_mean);
    }

    Ok(())
}
