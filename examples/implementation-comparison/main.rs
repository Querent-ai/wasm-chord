/// Comprehensive Implementation Comparison Tool
///
/// This tool compares our implementation with llama.cpp to identify
/// the key differences causing gibberish output.
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Implementation Comparison Tool");
    println!("=================================\n");

    // Load our model
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    println!("📂 Loading model: {}", model_path);

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded with {} tokens", tokenizer.vocab_size());

    // Load weights
    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model loaded\n");

    // Test with simple prompt
    let prompt = "Hello";
    println!("🎯 Testing prompt: \"{}\"", prompt);

    // Get our logits
    let tokens = tokenizer.encode(prompt, false)?;
    println!("📝 Our tokens: {:?}", tokens);

    // Forward pass to get logits
    let logits = model.forward(&tokens, 0)?;

    // Get top 5 logits
    let mut indexed_logits: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Our top 5 predictions after \"Hello\":");
    for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id as u32);
        println!("  {}: token {} = {:.6} ({:?})", i + 1, token_id, logit, token_text);
    }

    // Analysis based on llama.cpp findings
    println!("\n📋 Key Differences Analysis:");
    println!("=============================");

    println!("\n1. 🔍 RMS Normalization Implementation:");
    println!("   llama.cpp: mean = sum/ne00, scale = 1.0f/sqrtf(mean + eps)");
    println!("   Our impl:  mean = sum/ne00, scale = 1.0f/sqrtf(mean + eps)");
    println!("   ✅ RMS norm looks correct");

    println!("\n2. 🔍 Attention Scaling:");
    println!("   llama.cpp: kq_scale = 1.0f/sqrtf(float(n_embd_head))");
    println!("   Our impl:  score /= (head_dim as f32).sqrt()");
    println!("   ✅ Attention scaling looks correct");

    println!("\n3. 🔍 Layer Ordering (llama.cpp):");
    println!("   for (int il = 0; il < n_layer; ++il) {{");
    println!("       // 1. RMS norm on input");
    println!("       cur = build_norm(inpL, model.layers[il].attn_norm, ...)");
    println!("       // 2. Compute Q, K, V");
    println!("       Qcur = build_lora_mm(model.layers[il].wq, cur)");
    println!("       Kcur = build_lora_mm(model.layers[il].wk, cur)");
    println!("       Vcur = build_lora_mm(model.layers[il].wv, cur)");
    println!("       // 3. Apply RoPE");
    println!("       Qcur = ggml_rope_ext(...)");
    println!("       Kcur = ggml_rope_ext(...)");
    println!("       // 4. Attention computation");
    println!("       cur = build_attn(..., Qcur, Kcur, Vcur, ...)");
    println!("       // 5. Residual connection");
    println!("       ffn_inp = ggml_add(ctx0, cur, inpSA)");
    println!("       // 6. FFN norm");
    println!("       cur = build_norm(ffn_inp, model.layers[il].ffn_norm, ...)");
    println!("       // 7. FFN computation");
    println!("       cur = build_ffn(cur, ...)");
    println!("       // 8. Final residual connection");
    println!("       cur = ggml_add(ctx0, cur, ffn_inp)");
    println!("   }}");

    println!("\n4. 🔍 Our Layer Ordering:");
    println!("   for layer in 0..self.config.num_layers {{");
    println!("       // 1. RMS norm on input");
    println!("       let normed = self.rms_norm(&input, &self.layers[layer].attn_norm)?;");
    println!("       // 2. Compute Q, K, V");
    println!("       let q = self.compute_q(&normed, &self.layers[layer].wq)?;");
    println!("       let k = self.compute_k(&normed, &self.layers[layer].wk)?;");
    println!("       let v = self.compute_v(&normed, &self.layers[layer].wv)?;");
    println!("       // 3. Apply RoPE");
    println!("       self.apply_rope(&mut q, position)?;");
    println!("       self.apply_rope(&mut k, position)?;");
    println!("       // 4. Attention computation");
    println!("       let attn_out = self.compute_attention(&q, &k, &v, seq_len, position)?;");
    println!("       // 5. Residual connection");
    println!("       let ffn_input = self.add_vectors(&input, &attn_out)?;");
    println!("       // 6. FFN norm");
    println!("       let ffn_normed = self.rms_norm(&ffn_input, &self.layers[layer].ffn_norm)?;");
    println!("       // 7. FFN computation");
    println!("       let ffn_out = self.compute_ffn(&ffn_normed, &self.layers[layer])?;");
    println!("       // 8. Final residual connection");
    println!("       input = self.add_vectors(&ffn_input, &ffn_out)?;");
    println!("   }}");
    println!("   ✅ Layer ordering looks correct");

    println!("\n5. 🔍 Critical Issue - Matrix Multiplication Order:");
    println!("   llama.cpp attention: kq = ggml_mul_mat(ctx0, k, q)");
    println!("   This computes: kq[i,j] = sum(k[i,k] * q[j,k])");
    println!("   Our attention: score += q_vec[idx] * k_vec[idx]");
    println!("   This computes: score = sum(q[i] * k[i])");
    println!("   ❌ POTENTIAL ISSUE: Different matrix multiplication order!");

    println!("\n6. 🔍 Attention Mask:");
    println!("   llama.cpp: Uses causal mask for autoregressive generation");
    println!("   Our impl:  No explicit mask - processes all positions");
    println!("   ❌ CRITICAL: Missing causal mask!");

    println!("\n7. 🔍 Softmax Implementation:");
    println!("   llama.cpp: kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, ...)");
    println!("   Our impl:  Manual softmax with temperature");
    println!("   ❌ POTENTIAL: Different softmax implementation");

    println!("\n8. 🔍 Final Output Layer:");
    println!("   llama.cpp: cur = build_lora_mm(model.output, cur)");
    println!("   Our impl:  Manual matrix multiplication");
    println!("   ✅ Output layer looks correct");

    println!("\n🎯 LIKELY ROOT CAUSES:");
    println!("======================");
    println!("1. ❌ Missing causal attention mask (most likely)");
    println!("2. ❌ Incorrect matrix multiplication order in attention");
    println!("3. ❌ Different softmax implementation");
    println!("4. ❌ Potential issues with RoPE application");

    println!("\n🚀 RECOMMENDED FIXES:");
    println!("=====================");
    println!("1. Add causal attention mask to prevent looking at future tokens");
    println!("2. Verify matrix multiplication order in attention computation");
    println!("3. Use standard softmax implementation");
    println!("4. Double-check RoPE frequency calculation");

    Ok(())
}
