/// LM Head Debug Tool
/// This tool loads the model and debugs the LM head application step by step
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 LM Head Debug Tool");
    println!("====================\n");

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
    println!("   Config: vocab_size={}, hidden_size={}", config.vocab_size, config.hidden_size);

    // Test with "Hello"
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false)?;
    println!("\n📝 Input: '{}' → tokens: {:?}", prompt, tokens);

    // Get hidden states before LM head
    println!("\n🔍 Forward Pass Analysis:");
    println!("=========================");

    // We need to manually run the forward pass to inspect intermediate states
    let seq_len = tokens.len();
    let hidden_size = config.hidden_size;

    // 1. Token embeddings
    let mut hidden_states = vec![0.0; seq_len * hidden_size];
    for (i, &token_id) in tokens.iter().enumerate() {
        let start_idx = i * hidden_size;
        let end_idx = start_idx + hidden_size;
        hidden_states[start_idx..end_idx].copy_from_slice(
            &model.token_embeddings
                [(token_id as usize * hidden_size)..((token_id as usize + 1) * hidden_size)],
        );
    }

    println!("✅ Token embeddings applied");
    println!("   Hidden states shape: [seq_len={}, hidden_size={}]", seq_len, hidden_size);

    // Check hidden states stats
    let sum: f32 = hidden_states.iter().sum();
    let mean = sum / hidden_states.len() as f32;
    let max = hidden_states.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = hidden_states.iter().copied().fold(f32::INFINITY, f32::min);
    println!(
        "   Hidden states stats: sum={:.6}, mean={:.6}, min={:.6}, max={:.6}",
        sum, mean, min, max
    );

    // 2. Run through transformer layers
    for layer_idx in 0..config.num_layers {
        println!("\n🔄 Layer {}:", layer_idx);

        // Attention + FFN (simplified - we'll use the model's forward method)
        // For now, let's just check the layer weights
        let layer = &model.layers[layer_idx];
        println!("   Layer loaded: attention + FFN");
    }

    // 3. Final RMSNorm
    println!("\n📏 Final RMSNorm:");
    hidden_states = model.rms_norm(&hidden_states, &model.output_norm)?;

    let sum: f32 = hidden_states.iter().sum();
    let mean = sum / hidden_states.len() as f32;
    let max = hidden_states.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = hidden_states.iter().copied().fold(f32::INFINITY, f32::min);
    println!(
        "   After RMSNorm stats: sum={:.6}, mean={:.6}, min={:.6}, max={:.6}",
        sum, mean, min, max
    );

    // 4. LM Head Analysis
    println!("\n🎯 LM Head Analysis:");
    println!("===================");

    println!("   LM head shape: [hidden_size={}, vocab_size={}]", hidden_size, config.vocab_size);
    println!("   LM head total elements: {}", model.lm_head.len());
    println!("   Expected elements: {}", hidden_size * config.vocab_size);

    // Check LM head stats
    let lm_sum: f32 = model.lm_head.iter().sum();
    let lm_mean = lm_sum / model.lm_head.len() as f32;
    let lm_max = model.lm_head.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lm_min = model.lm_head.iter().copied().fold(f32::INFINITY, f32::min);
    println!(
        "   LM head stats: sum={:.6}, mean={:.6}, min={:.6}, max={:.6}",
        lm_sum, lm_mean, lm_min, lm_max
    );

    // Check first few LM head weights
    let preview_len = 10.min(model.lm_head.len());
    println!("   LM head preview: {:?}", &model.lm_head[..preview_len]);

    // 5. Manual LM head computation
    println!("\n🧮 Manual LM Head Computation:");
    println!("=============================");

    // For the first token position (seq_len=1)
    let first_hidden = &hidden_states[..hidden_size];
    println!("   First hidden state (first 10): {:?}", &first_hidden[..10.min(hidden_size)]);

    // Compute logits manually for a few tokens
    let test_tokens = vec![15043, 3869, 1939, 31982]; // Hello, Yes, No, 番
    let token_names = vec!["Hello", "Yes", "No", "番"];

    for (token_id, token_name) in test_tokens.iter().zip(token_names.iter()) {
        let mut logit = 0.0f32;
        for i in 0..hidden_size {
            let weight_idx = i * config.vocab_size + *token_id as usize;
            if weight_idx < model.lm_head.len() {
                logit += first_hidden[i] * model.lm_head[weight_idx];
            }
        }
        println!("   Manual logit for '{}' (token {}): {:.6}", token_name, token_id, logit);
    }

    // 6. Use model's forward method to get final logits
    println!("\n🔄 Model Forward Pass:");
    println!("=====================");

    let logits = model.forward(&tokens, 0)?;
    println!("   Model forward result length: {}", logits.len());
    println!("   Expected length: {}", config.vocab_size);

    // Check logits for our test tokens
    for (token_id, token_name) in test_tokens.iter().zip(token_names.iter()) {
        if *token_id < logits.len() {
            println!(
                "   Model logit for '{}' (token {}): {:.6}",
                token_name, token_id, logits[*token_id as usize]
            );
        }
    }

    // Find top token
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_token = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > max_logit {
            max_logit = logit;
            max_token = i;
        }
    }

    let token_text = tokenizer.id_to_token(max_token as u32);
    println!("\n🏆 Final Result:");
    println!("===============");
    println!(
        "   Top token: {} (id: {}, logit: {:.6})",
        token_text.unwrap_or("<unknown>"),
        max_token,
        max_logit
    );

    Ok(())
}
