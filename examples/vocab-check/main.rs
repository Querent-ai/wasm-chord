use wasm_chord_core::{GGUFParser, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Tokenizer ↔ Model Vocab Alignment Check");
    println!("==========================================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    let file = std::fs::File::open(model_path)?;
    let mut parser = GGUFParser::new(file);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: wasm_chord_runtime::TransformerConfig = config_data.into();

    let tokenizer = Tokenizer::from_gguf(&meta)?;

    // 1. Check vocab sizes
    println!("📊 VOCAB SIZE COMPARISON:");
    println!("  TOKENIZER vocab_size = {}", tokenizer.vocab_size());
    println!("  MODEL config.vocab_size = {}", config.vocab_size);

    if tokenizer.vocab_size() != config.vocab_size {
        println!("  ❌ MISMATCH! Tokenizer and model have different vocab sizes");
    } else {
        println!("  ✅ Vocab sizes match");
    }

    // 2. Decode the suspect tokens
    println!("\n🔍 SUSPECT TOKEN ANALYSIS:");
    for tok in [24081u32, 11032u32] {
        match tokenizer.decode(&[tok], true) {
            Ok(txt) => println!("  token id {} -> tokenizer.decode = {:?}", tok, txt),
            Err(e) => println!("  token id {} -> decode FAILED: {}", tok, e),
        }
    }

    // 3. Check a few other tokens for comparison
    println!("\n🔍 OTHER TOKEN SAMPLES:");
    for test_token in [1u32, 15043u32, 1000u32, 5000u32] {
        if test_token < tokenizer.vocab_size() as u32 {
            match tokenizer.decode(&[test_token], true) {
                Ok(txt) => println!("  token {} -> {:?}", test_token, txt),
                Err(_) => println!("  token {} -> <decode-fail>", test_token),
            }
        }
    }

    // 4. Check head dimensions
    println!("\n🔍 HEAD DIMENSION CHECK:");
    let head_dim = config.hidden_size / config.num_heads;
    println!(
        "  hidden={} heads={} head_dim={} num_kv_heads={}",
        config.hidden_size, config.num_heads, head_dim, config.num_kv_heads
    );

    if !config.hidden_size.is_multiple_of(config.num_heads) {
        println!("  ❌ head_dim is not integer!");
    } else {
        println!("  ✅ head_dim is integer");
    }

    // 5. Check if there's an output bias in the metadata
    println!("\n🔍 OUTPUT BIAS CHECK:");
    let mut found_bias = false;
    for tensor_desc in meta.tensors.iter() {
        if tensor_desc.name.contains("bias") {
            println!("  Found bias tensor: {}", tensor_desc.name);
            found_bias = true;
        }
    }
    if !found_bias {
        println!("  ℹ️  No bias tensors found (this is normal for many models)");
    }

    Ok(())
}
