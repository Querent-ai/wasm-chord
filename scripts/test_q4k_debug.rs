use wasm_chord_core::quant::{BlockQ4_K, get_scale_min_k4};
use std::fs::File;
use std::io::{BufReader, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a small Q4_K tensor from our model
    let mut file = File::open("models/tinyllama-1.1b.Q4_K_M.gguf")?;
    let mut reader = BufReader::new(file);
    
    // Skip to a Q4_K tensor
    let mut buffer = [0u8; 144]; // Q4_K block size
    reader.read_exact(&mut buffer)?;
    
    // Convert to BlockQ4_K
    let block: BlockQ4_K = unsafe { std::ptr::read(buffer.as_ptr() as *const _) };
    
    // Convert f16 scales to f32
    let d = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.dmin).to_f32();
    
    println!("Q4_K block: d={:.6}, min={:.6}", d, min);
    println!("Scales: {:?}", &block.scales[..12]);
    
    // Check first few scale extractions
    for i in 0..4 {
        let (sc, m) = get_scale_min_k4(i, &block.scales);
        println!("Scale {}: sc={}, m={}", i, sc, m);
        println!("  d1 = d * sc = {:.6} * {} = {:.6}", d, sc, d * sc as f32);
        println!("  m1 = min * m = {:.6} * {} = {:.6}", min, m, min * m as f32);
    }
    
    Ok(())
}
