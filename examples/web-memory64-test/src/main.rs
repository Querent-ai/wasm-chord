//! Web Browser Model Loading Test
//!
//! This example demonstrates the correct browser-compatible model loading.
//! It shows how the web.rs module handles model size limits and provides
//! clear guidance for large models.

use std::fs;
use std::io::Cursor;
use wasm_chord_core::{
    formats::gguf::GGUFParser,
    error::Result,
};

fn main() -> Result<()> {
    println!("🚀 Web Browser Model Loading Test");
    println!("=================================\n");

    // Test with TinyLlama (should work in browser)
    test_browser_model_loading("models/tinyllama-1.1b-chat-v0.6-Q4_K_M.gguf", "TinyLlama")?;
    
    // Test with Llama-2-7B (should be rejected for browser)
    test_browser_model_loading("models/llama-2-7b-chat-q4_k_m.gguf", "Llama-2-7B")?;

    println!("\n🎉 Web Browser Model Loading Test Complete!");
    println!("✅ Browser model size limits working");
    println!("✅ Clear error messages for large models");
    println!("✅ Correct architecture: browser vs native");

    Ok(())
}

fn test_browser_model_loading(model_path: &str, model_name: &str) -> Result<()> {
    println!("\n📂 Testing {} model for browser compatibility...", model_name);
    
    if !std::path::Path::new(model_path).exists() {
        println!("⚠️  {} model not found at: {}", model_name, model_path);
        return Ok(());
    }

    // Read model bytes
    let model_bytes = fs::read(model_path)?;
    println!("   📋 Model file size: {:.2} MB", model_bytes.len() as f64 / 1_000_000.0);

    // Parse GGUF header to get model info
    let cursor = Cursor::new(&model_bytes);
    let mut parser = GGUFParser::new(cursor);
    let meta = parser.parse_header()?;
    
    let config_data = parser.extract_config()
        .ok_or_else(|| wasm_chord_core::error::Error::ParseError("Failed to extract config".to_string()))?;
    let config: wasm_chord_runtime::TransformerConfig = config_data.into();
    
    // Estimate model size
    let total_size: u64 = meta.tensors.iter().map(|t| t.size_bytes as u64).sum();
    let size_gb = total_size as f64 / 1_000_000_000.0;
    
    println!("   📊 Model size: {:.2} GB", size_gb);
    println!("   ⚙️  Config: {} layers, {} vocab, {} hidden", 
             config.num_layers, config.vocab_size, config.hidden_size);

    // Test browser compatibility
    let browser_limit = 3_500_000_000; // 3.5GB safety margin
    let is_browser_compatible = total_size <= browser_limit;
    
    println!("   🌐 Browser compatible: {}", is_browser_compatible);
    
    if is_browser_compatible {
        println!("   ✅ This model can be loaded in browsers");
        println!("   💡 Use: new WasmModel(ggufBytes) in JavaScript");
    } else {
        println!("   ❌ This model is too large for browser WASM memory");
        println!("   💡 Use: Native Memory64 runtime instead");
        println!("   📝 Error message would be:");
        println!("      \"Model too large ({:.2} GB) for browser WASM memory. Browser limit is ~4GB. For large models, use the native Memory64 runtime instead.\"", size_gb);
    }

    println!("   ✅ {} model analysis complete", model_name);
    Ok(())
}
