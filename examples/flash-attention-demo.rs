//! Flash Attention Demo
//!
//! Demonstrates using Flash Attention for memory-efficient inference.

use wasm_chord_runtime::{
    attention::{AttentionBackend, create_attention, Attention},
};

fn main() {
    println!("🔥 Flash Attention Demo\n");
    
    // Test configuration
    let batch_size = 1;
    let num_heads = 8;
    let seq_len_q = 512;
    let seq_len_k = 512;
    let head_dim = 64;
    
    // Create attention implementations
    let standard = create_attention(AttentionBackend::Standard);
    let flash = create_attention(AttentionBackend::Flash);
    
    println!("📊 Configuration:");
    println!("   Batch: {}, Heads: {}, SeqLen: {}, HeadDim: {}", 
             batch_size, num_heads, seq_len_q, head_dim);
    println!();
    
    // Memory comparison
    let standard_mem = standard.estimated_memory(seq_len_q, head_dim, num_heads);
    let flash_mem = flash.estimated_memory(seq_len_q, head_dim, num_heads);
    
    println!("💾 Memory Usage:");
    println!("   Standard Attention: {} MB", standard_mem / 1_000_000);
    println!("   Flash Attention:    {} MB", flash_mem / 1_000_000);
    println!("   Reduction:          {}x", standard_mem / flash_mem.max(1));
    println!();
    
    // Create dummy data
    let q = vec![0.1; batch_size * num_heads * seq_len_q * head_dim];
    let k = vec![0.2; batch_size * num_heads * seq_len_k * head_dim];
    let v = vec![0.3; batch_size * num_heads * seq_len_k * head_dim];
    
    // Run both implementations
    println!("⚡ Running inference...");
    
    let start = std::time::Instant::now();
    let _standard_output = standard.forward(&q, &k, &v, None, batch_size, num_heads, seq_len_q, seq_len_k, head_dim)
        .expect("Standard attention failed");
    let standard_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let _flash_output = flash.forward(&q, &k, &v, None, batch_size, num_heads, seq_len_q, seq_len_k, head_dim)
        .expect("Flash attention failed");
    let flash_time = start.elapsed();
    
    println!("   Standard: {:?}", standard_time);
    println!("   Flash:    {:?}", flash_time);
    println!("   Speedup:  {:.2}x", standard_time.as_secs_f64() / flash_time.as_secs_f64());
    println!();
    
    println!("✅ Flash Attention demo complete!");
    println!();
    println!("💡 Key Benefits:");
    println!("   • {}x less memory", standard_mem / flash_mem.max(1));
    println!("   • Enables longer sequences");
    println!("   • Scales to GPU with CUDA/Metal");
}

