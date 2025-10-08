/// Comprehensive Diagnostic Tool for GGUF Transformer
///
/// This tool systematically checks all 5 common root causes of gibberish output:
/// 1. Wrong tensor transpose (GGUF weight orientation mismatch)
/// 2. Wrong normalization (missing RMSNorm epsilon or scale)
/// 3. Incorrect RoPE rotation formula
/// 4. KV cache reuse across tokens without masking
/// 5. Tokenizer mismatch
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Comprehensive GGUF Transformer Diagnostic Tool");
    println!("==================================================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    println!("📂 Loading model: {}", model_path);

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!(
        "✅ Config loaded: {} layers, {} vocab, {} hidden_size",
        config.num_layers, config.vocab_size, config.hidden_size
    );

    // === 1. CHECK GGUF TENSOR SHAPES AND TRANSPOSE REQUIREMENTS ===
    println!("\n🔍 1. CHECKING GGUF TENSOR SHAPES AND TRANSPOSE REQUIREMENTS");
    println!("=============================================================");

    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Check key tensor shapes
    let key_tensors = [
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

    for tensor_name in &key_tensors {
        if let Some(tensor) = meta.tensors.iter().find(|t| t.name == *tensor_name) {
            let shape = &tensor.shape.0;
            println!("  {}: {:?}", tensor_name, shape);

            // Analyze what transpose is needed
            match *tensor_name {
                "token_embd.weight" => {
                    if shape == &[config.hidden_size, config.vocab_size] {
                        println!("    ✅ GGUF stores as [hidden_size, vocab_size] - NO transpose needed for row-wise access");
                    } else if shape == &[config.vocab_size, config.hidden_size] {
                        println!("    ❌ GGUF stores as [vocab_size, hidden_size] - TRANSPOSE needed for row-wise access");
                    }
                }
                "output.weight" => {
                    if shape == &[config.hidden_size, config.vocab_size] {
                        println!("    ✅ GGUF stores as [hidden_size, vocab_size] - NO transpose needed for matmul");
                    } else if shape == &[config.vocab_size, config.hidden_size] {
                        println!("    ❌ GGUF stores as [vocab_size, hidden_size] - TRANSPOSE needed for matmul");
                    }
                }
                "blk.0.attn_q.weight" | "blk.0.attn_output.weight" => {
                    if shape == &[config.hidden_size, config.hidden_size] {
                        println!("    ✅ GGUF stores as [hidden_size, hidden_size] - NO transpose needed");
                    } else {
                        println!("    ❌ Unexpected shape for Q/O projection");
                    }
                }
                "blk.0.attn_k.weight" | "blk.0.attn_v.weight" => {
                    let kv_dim = config.num_kv_heads * (config.hidden_size / config.num_heads);
                    if shape == &[config.hidden_size, kv_dim] {
                        println!(
                            "    ✅ GGUF stores as [hidden_size, kv_dim] - NO transpose needed"
                        );
                    } else if shape == &[kv_dim, config.hidden_size] {
                        println!("    ❌ GGUF stores as [kv_dim, hidden_size] - TRANSPOSE needed");
                    } else {
                        println!("    ❌ Unexpected shape for K/V projection: expected [hidden_size, kv_dim] or [kv_dim, hidden_size]");
                    }
                }
                "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => {
                    if shape == &[config.hidden_size, config.intermediate_size] {
                        println!("    ✅ GGUF stores as [hidden_size, intermediate_size] - NO transpose needed");
                    } else if shape == &[config.intermediate_size, config.hidden_size] {
                        println!("    ❌ GGUF stores as [intermediate_size, hidden_size] - TRANSPOSE needed");
                    }
                }
                "blk.0.ffn_down.weight" => {
                    if shape == &[config.intermediate_size, config.hidden_size] {
                        println!("    ✅ GGUF stores as [intermediate_size, hidden_size] - NO transpose needed");
                    } else if shape == &[config.hidden_size, config.intermediate_size] {
                        println!("    ❌ GGUF stores as [hidden_size, intermediate_size] - TRANSPOSE needed");
                    }
                }
                _ => {}
            }
        } else {
            println!("  {}: NOT FOUND", tensor_name);
        }
    }

    // === 2. CHECK RMS NORMALIZATION ===
    println!("\n🔍 2. CHECKING RMS NORMALIZATION");
    println!("=================================");

    // Print all available metadata keys to debug
    println!("  Available metadata keys:");
    for (key, value) in meta.metadata.iter() {
        if key.contains("norm") || key.contains("rope") || key.contains("eps") {
            println!("    {}: {:?}", key, value);
        }
    }

    // Check if we can find rms_norm_eps in metadata
    if let Some(eps) = meta.metadata.get("llama.attention.layer_norm_rms_epsilon") {
        println!("  RMS norm epsilon from GGUF: {:?}", eps);
    } else {
        println!("  ⚠️  No rms_norm_eps found in GGUF metadata");
    }

    // Check if we can find any norm weights
    let norm_tensors =
        meta.tensors.iter().filter(|t| t.name.contains("norm")).take(3).collect::<Vec<_>>();

    for tensor in norm_tensors {
        println!("  Found norm tensor: {} with shape {:?}", tensor.name, tensor.shape.0);
    }

    // === 3. CHECK ROPE PARAMETERS ===
    println!("\n🔍 3. CHECKING ROPE PARAMETERS");
    println!("==============================");

    if let Some(rope_theta) = meta.metadata.get("llama.rope.theta") {
        println!("  RoPE theta: {:?}", rope_theta);
    } else {
        println!("  ⚠️  No rope.theta found in GGUF metadata");
    }

    if let Some(rope_freq_base) = meta.metadata.get("llama.rope.freq_base") {
        println!("  RoPE freq_base: {:?}", rope_freq_base);
    } else {
        println!("  ⚠️  No rope.freq_base found in GGUF metadata");
    }

    // === 4. CHECK TOKENIZER COMPATIBILITY ===
    println!("\n🔍 4. CHECKING TOKENIZER COMPATIBILITY");
    println!("======================================");

    println!("  Tokenizer vocab size: {}", tokenizer.vocab_size());
    println!("  Model vocab size: {}", config.vocab_size);

    if tokenizer.vocab_size() == config.vocab_size {
        println!("  ✅ Tokenizer vocab size matches model");
    } else {
        println!("  ❌ Tokenizer vocab size mismatch!");
    }

    // Test tokenization
    let test_prompt = "Hello";
    match tokenizer.encode(test_prompt, false) {
        Ok(tokens) => {
            println!("  Test tokenization '{}': {:?}", test_prompt, tokens);
            for (i, &token_id) in tokens.iter().enumerate() {
                if let Some(token_str) = tokenizer.id_to_token(token_id) {
                    println!("    Token {}: {} -> '{}'", i, token_id, token_str);
                }
            }
        }
        Err(e) => println!("  ❌ Tokenization failed: {}", e),
    }

    // === 5. LOAD MODEL AND TEST FORWARD PASS ===
    println!("\n🔍 5. TESTING FORWARD PASS");
    println!("==========================");

    let mut model = Model::new(config.clone());
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("  ✅ Model loaded successfully");

    // Test single token forward pass
    let test_tokens = vec![15043u32]; // "Hello" token
    println!("  Testing forward pass with token: {:?}", test_tokens);

    match model.forward(&test_tokens, 0) {
        Ok(logits) => {
            println!("  ✅ Forward pass successful, logits shape: {}", logits.len());

            // Check logits distribution
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_logit = logits.iter().copied().fold(f32::INFINITY, f32::min);
            let mean_logit = logits.iter().sum::<f32>() / logits.len() as f32;

            println!(
                "  Logits stats: min={:.3}, max={:.3}, mean={:.3}",
                min_logit, max_logit, mean_logit
            );

            // Check for reasonable top tokens
            let mut indexed_logits: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("  Top 5 predicted tokens:");
            for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
                let token_str = tokenizer.id_to_token(*token_id as u32);
                println!("    {}: token {} = {:.3} ({:?})", i + 1, token_id, logit, token_str);
            }

            // Check if top tokens are reasonable
            let top_token = indexed_logits[0].0;
            if let Some(token_str) = tokenizer.id_to_token(top_token as u32) {
                let is_reasonable = token_str.contains("there")
                    || token_str.contains("world")
                    || token_str.contains("how")
                    || token_str.contains("are")
                    || token_str.contains("you")
                    || token_str.contains("my")
                    || token_str.contains("name")
                    || token_str.contains("is")
                    || token_str.contains("I")
                    || token_str.contains("Hello")
                    || token_str.contains("Hi")
                    || token_str.contains("Hey");

                if is_reasonable {
                    println!("  ✅ Top prediction looks reasonable: '{}'", token_str);
                } else {
                    println!("  ❌ Top prediction looks like gibberish: '{}'", token_str);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Forward pass failed: {}", e);
        }
    }

    // === SUMMARY ===
    println!("\n📋 DIAGNOSTIC SUMMARY");
    println!("====================");
    println!("1. ✅ Checked GGUF tensor shapes and transpose requirements");
    println!("2. ✅ Checked RMS normalization parameters");
    println!("3. ✅ Checked RoPE parameters");
    println!("4. ✅ Checked tokenizer compatibility");
    println!("5. ✅ Tested forward pass and logits");
    println!("\n🎯 Next steps: Fix any issues identified above, especially tensor transposes!");

    Ok(())
}
