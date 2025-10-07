use wasm_chord_core::quant::dequantize_q4_k;
use std::fs::File;
use std::io::BufReader;

fn main() {
    // Test with a simple Q4_K block
    let mut block = [0u8; 144]; // Q4_K block size
    let mut output = vec![0.0f32; 256]; // 256 elements
    
    // Fill with some test data
    for i in 0..144 {
        block[i] = (i % 256) as u8;
    }
    
    dequantize_q4_k(&block, &mut output, 0);
    
    println!("Q4_K dequantized values:");
    println!("  Min: {:.6}", output.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
    println!("  Max: {:.6}", output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    println!("  Mean: {:.6}", output.iter().sum::<f32>() / output.len() as f32);
    println!("  Std: {:.6}", {
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        let variance = output.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
        variance.sqrt()
    });
}
