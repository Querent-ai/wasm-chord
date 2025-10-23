/// Test dequantization of the first block from the model

use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::formats::gguf::GGUFParser;
use wasm_chord_core::quant::{dequantize_q4_k, BlockQ4_K, QK_K};
use wasm_chord_core::tensor::DataType;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    let tensors = parser.metadata().unwrap().tensors.clone();

    // Find the test tensor
    let test_tensor = tensors.iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight")
        .expect("Test tensor not found");

    println!("Found tensor: {}", test_tensor.name);
    println!("Shape: {:?}", test_tensor.shape);
    println!("Dtype: {:?}", test_tensor.dtype);

    // Get tensor data section offset (GGUF tensor offsets are relative!)
    let data_offset = parser.tensor_data_offset()
        .expect("Failed to get tensor data offset");
    let absolute_offset = data_offset + test_tensor.offset;

    println!("Data offset: {}, Tensor offset: {}, Absolute: {}",
             data_offset, test_tensor.offset, absolute_offset);

    // Read the raw data from correct location
    let raw_data = parser.read_tensor_data(absolute_offset, test_tensor.size_bytes)?;

    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_K>();
    println!("\nBlockQ4_K size: {} bytes", BLOCK_SIZE);
    println!("First block bytes (hex):");
    for (i, byte) in raw_data.iter().take(BLOCK_SIZE).enumerate() {
        if i % 16 == 0 {
            print!("\n{:04x}: ", i);
        }
        print!("{:02x} ", byte);
    }
    println!("\n");

    // Read first block
    let first_block = unsafe {
        std::ptr::read(raw_data.as_ptr() as *const BlockQ4_K)
    };

    println!("First block structure:");
    println!("  d (raw u16): 0x{:04x}", first_block.d);
    println!("  dmin (raw u16): 0x{:04x}", first_block.dmin);
    println!("  d (f16 → f32): {}", half::f16::from_bits(first_block.d).to_f32());
    println!("  dmin (f16 → f32): {}", half::f16::from_bits(first_block.dmin).to_f32());
    println!("  scales: {:02x?}", &first_block.scales);
    println!("  qs[0..16]: {:02x?}", &first_block.qs[0..16]);

    // Dequantize
    let mut output = vec![0.0f32; QK_K];
    dequantize_q4_k(&first_block, &mut output)?;

    println!("\nDequantized values (first 20):");
    for (i, val) in output.iter().take(20).enumerate() {
        println!("  output[{}] = {:.6}", i, val);
    }

    // Check for NaN/Inf
    let num_nan = output.iter().filter(|x| x.is_nan()).count();
    let num_inf = output.iter().filter(|x| x.is_infinite()).count();
    println!("\nNaN count: {}", num_nan);
    println!("Inf count: {}", num_inf);

    // Check value ranges
    let min_val = output.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!("Min value: {:.6}", min_val);
    println!("Max value: {:.6}", max_val);

    Ok(())
}
