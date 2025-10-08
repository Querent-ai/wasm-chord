/// Dump GGUF tensor shapes for debugging
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::GGUFParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    println!("📂 Analyzing model: {}\n", model_path);

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    // Print relevant tensor shapes
    let tensors_of_interest = vec![
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];

    for tensor_name in tensors_of_interest {
        if let Some(tensor) = meta.tensors.iter().find(|t| t.name == tensor_name) {
            println!(
                "{:<30} shape: {:?}  (n_elements: {})",
                tensor_name,
                tensor.shape.0,
                tensor.shape.0.iter().product::<usize>()
            );

            // Calculate what we expect
            if tensor_name == "blk.0.attn_q.weight" {
                // For Q: hidden_size × hidden_size
                // TinyLlama: 2048 × 2048 = 4194304
                println!("  → Expected for Q: [hidden_size, hidden_size] = [2048, 2048]");
                println!("  → Actual dimensions: {:?}", tensor.shape.0);
                println!("  → GGUF format: dimensions are [dim0, dim1, ...]");
                println!("  → For matmul X @ W: X is [seq_len, hidden_size], W is [hidden_size, hidden_size]");
                println!("  → If GGUF stores [2048, 2048], it's [in_features, out_features] - NO transpose!");
                println!();
            } else if tensor_name == "blk.0.attn_k.weight" {
                // For K: hidden_size × kv_dim (2048 × 256 = 524288)
                println!("  → Expected for K: [hidden_size, kv_dim] = [2048, 256]");
                println!("  → Actual dimensions: {:?}", tensor.shape.0);
                println!();
            }
        } else {
            println!("{:<30} NOT FOUND", tensor_name);
        }
    }

    Ok(())
}
