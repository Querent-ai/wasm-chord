/// Test RMSNorm implementation
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing RMSNorm Implementation");
    println!("===============================\n");

    // Create a minimal config
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 8,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_size: 16,
        max_seq_len: 128,
        num_layers: 1,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    };

    let model = Model::new(config);

    // Test input
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    println!("Input: {:?}", input);
    println!("Weight: {:?}", weight);

    // Test RMSNorm
    let result = model.rms_norm(&input, &weight)?;
    println!("RMSNorm result: {:?}", result);

    // Check if result is reasonable
    let sum: f32 = result.iter().sum();
    let mean = sum / result.len() as f32;
    println!("Sum: {:.6}, Mean: {:.6}", sum, mean);

    // Test with different weights
    let weight2 = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let result2 = model.rms_norm(&input, &weight2)?;
    println!("RMSNorm with weight=2: {:?}", result2);

    // Test with zero input
    let zero_input = vec![0.0; 8];
    let result3 = model.rms_norm(&zero_input, &weight)?;
    println!("RMSNorm with zero input: {:?}", result3);

    Ok(())
}
