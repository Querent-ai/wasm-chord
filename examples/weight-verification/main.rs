/// Weight Loading & Dequantization Verification Tool
/// This tool loads weights, dequantizes them manually, and compares with model forward pass
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Weight Loading & Dequantization Verification");
    println!("===============================================\n");

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    // Load model
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    let mut model = Model::new(config.clone());

    // Load weights
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

    println!("✅ Model loaded successfully");
    println!(
        "   Config: vocab_size={}, hidden_size={}, num_layers={}",
        config.vocab_size, config.hidden_size, config.num_layers
    );

    // Test with simple token
    let test_token = "Hello";
    let tokens = tokenizer.encode(test_token, false)?;
    println!("\n🔍 Testing with token: '{}' -> {:?}", test_token, tokens);

    // 1. Manual token embedding lookup
    println!("\n1️⃣ Manual Token Embedding Verification:");
    println!("========================================");

    let token_id = tokens[0] as usize;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;

    println!("   Token ID: {}", token_id);
    println!("   Hidden size: {}", hidden_size);
    println!("   Vocab size: {}", vocab_size);

    // Manual embedding extraction
    let mut manual_embedding = vec![0.0; hidden_size];
    for dim_idx in 0..hidden_size {
        let emb_idx = dim_idx * vocab_size + token_id;
        if emb_idx < model.token_embeddings.len() {
            manual_embedding[dim_idx] = model.token_embeddings[emb_idx];
        }
    }

    println!("   Manual embedding (first 10): {:?}", &manual_embedding[..10.min(hidden_size)]);
    println!(
        "   Manual embedding stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        manual_embedding.iter().sum::<f32>() / hidden_size as f32,
        compute_std(&manual_embedding),
        manual_embedding.iter().copied().fold(f32::INFINITY, f32::min),
        manual_embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // 2. Manual LM head computation
    println!("\n2️⃣ Manual LM Head Verification:");
    println!("================================");

    // Get hidden states from model forward pass
    let logits = model.forward(&tokens, 0)?;
    println!("   Model forward logits (first 10): {:?}", &logits[..10.min(logits.len())]);

    // Manual LM head computation
    let mut manual_logits = vec![0.0; vocab_size];
    for token_idx in 0..vocab_size {
        let mut sum = 0.0;
        for dim_idx in 0..hidden_size {
            let weight_idx = dim_idx * vocab_size + token_idx;
            if weight_idx < model.lm_head.len() {
                sum += manual_embedding[dim_idx] * model.lm_head[weight_idx];
            }
        }
        manual_logits[token_idx] = sum;
    }

    println!("   Manual logits (first 10): {:?}", &manual_logits[..10.min(vocab_size)]);
    println!(
        "   Manual logits stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        manual_logits.iter().sum::<f32>() / vocab_size as f32,
        compute_std(&manual_logits),
        manual_logits.iter().copied().fold(f32::INFINITY, f32::min),
        manual_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // 3. Compare manual vs model logits
    println!("\n3️⃣ Manual vs Model Logits Comparison:");
    println!("=====================================");

    let mut max_diff: f32 = 0.0;
    let mut diff_count = 0;
    for i in 0..vocab_size.min(logits.len()) {
        let diff = (manual_logits[i] - logits[i]).abs();
        if diff > 1e-5 {
            diff_count += 1;
            max_diff = max_diff.max(diff);
        }
    }

    println!("   Max difference: {:.6}", max_diff);
    println!("   Different values: {}/{}", diff_count, vocab_size.min(logits.len()));

    if max_diff < 1e-5 {
        println!("   ✅ Manual and model logits match!");
    } else {
        println!("   ❌ Manual and model logits differ significantly!");
    }

    // 4. Check specific token logits
    println!("\n4️⃣ Specific Token Logits:");
    println!("==========================");

    let test_tokens =
        vec![("Hello", token_id), ("Yes", 3869), ("No", 1939), ("番", 31982), ("rique", 29871)];

    for (token_name, token_idx) in test_tokens {
        if token_idx < manual_logits.len() && token_idx < logits.len() {
            println!(
                "   '{}' ({}): manual={:.6}, model={:.6}, diff={:.6}",
                token_name,
                token_idx,
                manual_logits[token_idx],
                logits[token_idx],
                (manual_logits[token_idx] - logits[token_idx]).abs()
            );
        }
    }

    // 5. Check weight statistics
    println!("\n5️⃣ Weight Statistics:");
    println!("===================");

    println!("   Token embeddings:");
    println!(
        "     Mean: {:.6}, Std: {:.6}, Min: {:.6}, Max: {:.6}",
        model.token_embeddings.iter().sum::<f32>() / model.token_embeddings.len() as f32,
        compute_std(&model.token_embeddings),
        model.token_embeddings.iter().copied().fold(f32::INFINITY, f32::min),
        model.token_embeddings.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    println!("   LM head:");
    println!(
        "     Mean: {:.6}, Std: {:.6}, Min: {:.6}, Max: {:.6}",
        model.lm_head.iter().sum::<f32>() / model.lm_head.len() as f32,
        compute_std(&model.lm_head),
        model.lm_head.iter().copied().fold(f32::INFINITY, f32::min),
        model.lm_head.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // 6. Check for NaN/Inf values
    println!("\n6️⃣ NaN/Inf Check:");
    println!("=================");

    let token_nan = model.token_embeddings.iter().filter(|&&x| x.is_nan()).count();
    let token_inf = model.token_embeddings.iter().filter(|&&x| x.is_infinite()).count();
    let lm_nan = model.lm_head.iter().filter(|&&x| x.is_nan()).count();
    let lm_inf = model.lm_head.iter().filter(|&&x| x.is_infinite()).count();

    println!("   Token embeddings: {} NaN, {} Inf", token_nan, token_inf);
    println!("   LM head: {} NaN, {} Inf", lm_nan, lm_inf);

    if token_nan > 0 || token_inf > 0 || lm_nan > 0 || lm_inf > 0 {
        println!("   ❌ Found NaN/Inf values in weights!");
    } else {
        println!("   ✅ No NaN/Inf values found in weights");
    }

    // 7. Test with different tokens
    println!("\n7️⃣ Multi-Token Test:");
    println!("====================");

    let test_prompts = vec!["Hello", "The", "Once", "if"];

    for prompt in test_prompts {
        let tokens = tokenizer.encode(prompt, false)?;
        let logits = model.forward(&tokens, 0)?;

        let mut max_logit = f32::NEG_INFINITY;
        let mut max_token = 0;
        for (i, &logit) in logits.iter().enumerate() {
            if logit > max_logit {
                max_logit = logit;
                max_token = i;
            }
        }

        let predicted_token = tokenizer.id_to_token(max_token as u32);
        let predicted_text = predicted_token.unwrap_or("<unknown>");

        println!("   '{}' -> '{}' (logit: {:.6})", prompt, predicted_text, max_logit);
    }

    println!("\n🏁 Weight verification complete!");

    Ok(())
}

fn compute_std(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}
