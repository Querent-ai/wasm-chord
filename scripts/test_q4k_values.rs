use wasm_chord_core::quant::{dequantize_q4_k, BlockQ4_K};
use std::fs::File;
use std::io::{BufReader, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a small Q4_K tensor from our model
    let mut file = File::open("models/tinyllama-1.1b.Q4_K_M.gguf")?;
    let mut reader = BufReader::new(file);
    
    // Skip to a Q4_K tensor (this is a hack, but let's see what we get)
    let mut buffer = [0u8; 144]; // Q4_K block size
    reader.read_exact(&mut buffer)?;
    
    // Convert to BlockQ4_K
    let block: BlockQ4_K = unsafe { std::ptr::read(buffer.as_ptr() as *const _) };
    
    // Dequantize
    let mut output = vec![0.0f32; 256]; // QK_K = 256
    dequantize_q4_k(&block, &mut output)?;
    
    println!("Q4_K dequantized values:");
    println!("  Min: {:.6}", output.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
    println!("  Max: {:.6}", output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    println!("  Mean: {:.6}", output.iter().sum::<f32>() / output.len() as f32);
    println!("  Std: {:.6}", {
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        let variance = output.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
        variance.sqrt()
    });
    
    // Show first 10 values
    println!("  First 10: {:?}", &output[..10]);
    
    Ok(())
}
