/// Check all blocks for NaN/Inf during dequantization

use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::formats::gguf::GGUFParser;
use wasm_chord_core::quant::{dequantize_q4_k, BlockQ4_K, QK_K};

fn main() -> anyhow::Result<()> {
    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    let tensors = parser.metadata().unwrap().tensors.clone();
    let test_tensor = tensors.iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight")
        .expect("Test tensor not found");

    println!("Testing tensor: {}", test_tensor.name);
    println!("Shape: {:?}", test_tensor.shape);

    // Get tensor data section offset (GGUF tensor offsets are relative!)
    let data_offset = parser.tensor_data_offset()
        .expect("Failed to get tensor data offset");

    println!("Data section offset: {} bytes", data_offset);
    println!("Tensor relative offset: {} bytes", test_tensor.offset);

    // Calculate absolute offset
    let absolute_offset = data_offset + test_tensor.offset;
    println!("Absolute offset: {} bytes\n", absolute_offset);

    // Read raw data from correct location
    let raw_data = parser.read_tensor_data(absolute_offset, test_tensor.size_bytes)?;

    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_K>();
    let num_blocks = raw_data.len() / BLOCK_SIZE;
    println!("Total blocks: {}", num_blocks);

    // Check each block
    let mut bad_blocks = Vec::new();
    let mut total_nan = 0;
    let mut total_inf = 0;

    for (block_idx, block_data) in raw_data.chunks_exact(BLOCK_SIZE).enumerate() {
        let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_K) };

        // Try to dequantize
        let mut output = vec![0.0f32; QK_K];
        dequantize_q4_k(&block, &mut output)?;

        // Check for NaN/Inf
        let num_nan = output.iter().filter(|x| x.is_nan()).count();
        let num_inf = output.iter().filter(|x| x.is_infinite()).count();

        if num_nan > 0 || num_inf > 0 {
            bad_blocks.push((block_idx, num_nan, num_inf));
            total_nan += num_nan;
            total_inf += num_inf;

            if bad_blocks.len() <= 10 {
                println!("\nBlock {}: {} NaN, {} Inf", block_idx, num_nan, num_inf);
                println!("  d (raw): 0x{:04x} = {}", block.d, half::f16::from_bits(block.d).to_f32());
                println!("  dmin (raw): 0x{:04x} = {}", block.dmin, half::f16::from_bits(block.dmin).to_f32());
                println!("  First few values: {:?}", &output[0..10.min(output.len())]);
            }
        }
    }

    println!("\n========== Summary ==========");
    println!("Total bad blocks: {} / {}", bad_blocks.len(), num_blocks);
    println!("Total NaN values: {}", total_nan);
    println!("Total Inf values: {}", total_inf);

    if bad_blocks.len() > 10 {
        println!("\nFirst 10 bad blocks: {:?}", &bad_blocks[..10]);
        println!("Last 10 bad blocks: {:?}", &bad_blocks[bad_blocks.len()-10..]);
    }

    Ok(())
}
