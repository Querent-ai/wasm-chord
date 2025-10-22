/// Direct comparison with Ollama TinyLlama
/// This loads the exact same model and compares first token generation
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Direct Comparison with Ollama TinyLlama");
    println!("==========================================\n");

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    // Test 1: Load model and check config
    println!("1️⃣ Loading Model Configuration");
    println!("-------------------------------");

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    println!("✅ GGUF file loaded successfully");
    println!("   Architecture: {:?}", meta.architecture);
    println!("   Tensor count: {}", meta.tensor_count);
    println!("   Vocab size: {:?}", meta.vocab_size);

    // Extract config from GGUF
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    println!("✅ Model configuration:");
    println!("   Vocab size: {}", config.vocab_size);
    println!("   Hidden size: {}", config.hidden_size);
    println!("   Num layers: {}", config.num_layers);
    println!("   Num heads: {}", config.num_heads);
    println!("   Num KV heads: {}", config.num_kv_heads);
    println!("   Intermediate size: {}", config.intermediate_size);
    println!("   RMS norm eps: {}", config.rms_norm_eps);
    println!("   RoPE theta: {}", config.rope_theta);

    // Test 2: Load tokenizer
    println!("\n2️⃣ Loading Tokenizer");
    println!("--------------------");

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Test 3: Tokenize "Hello" with detailed debugging
    println!("\n3️⃣ Tokenization Analysis");
    println!("-------------------------");

    let prompt = "Hello";
    println!("🔍 Encoding prompt: {:?}", prompt);

    let tokens = tokenizer.encode(prompt, true)?; // Add BOS token!
    println!("✅ Tokens for '{}': {:?}", prompt, tokens);

    // Check what token ID 15043 maps to (expected from Ollama)
    if let Some(token_text) = tokenizer.id_to_token(15043) {
        println!("✅ Token ID 15043 maps to: {:?}", token_text);
    } else {
        println!("❌ Token ID 15043 not found in vocabulary");
    }

    // Check what our tokenized result maps to
    if let Some(token_text) = tokenizer.id_to_token(tokens[0]) {
        println!("✅ Our token {} maps to: {:?}", tokens[0], token_text);
    }

    // Test multiple prompts to understand tokenization patterns
    println!("\n🔍 Testing multiple prompts:");
    let test_prompts = vec!["Hello", "Yes", "No", "The", "A", "Paris"];
    for prompt in test_prompts {
        if let Ok(tokens) = tokenizer.encode(prompt, false) {
            let token_text = tokenizer.id_to_token(tokens[0]).unwrap_or("<unknown>");
            println!("  '{}' -> token {} ({:?})", prompt, tokens[0], token_text);
        }
    }

    // Check vocabulary structure
    println!("\n🔍 Vocabulary sample (first 20 tokens):");
    for i in 0..20.min(tokenizer.vocab_size()) {
        if let Some(token_text) = tokenizer.id_to_token(i as u32) {
            println!("  {}: {:?}", i, token_text);
        }
    }

    // Test 4: Load model weights
    println!("\n4️⃣ Loading Model Weights");
    println!("-------------------------");

    let mut model = Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    // Register all tensors
    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Reopen file for loading
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    // Load weights
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model weights loaded successfully");

    // Test 5: Generate first token with greedy sampling
    println!("\n5️⃣ Generating First Token (Greedy)");
    println!("-----------------------------------");

    let logits = model.forward(&tokens, 0)?;
    println!("✅ Forward pass completed");
    println!("   Logits shape: {}", logits.len());

    // Find top tokens
    let mut indexed_logits: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("✅ Top 20 tokens:");
    for (rank, (token_id, logit)) in indexed_logits.iter().take(20).enumerate() {
        let token_text = tokenizer.id_to_token(*token_id as u32);
        let token_str = token_text.unwrap_or("<unknown>");
        println!("   {}: {} (id: {}, logit: {:.6})", rank + 1, token_str, token_id, logit);
    }

    // Check specific tokens that might be expected
    let test_tokens = vec![
        ("How", vec!["How", "▁How"]),
        ("there", vec!["there", "▁there"]),
        (" there", vec!["▁there"]),
        ("!", vec!["!"]),
        ("Yes", vec!["Yes", "▁Yes"]), // Added Yes token for debugging
        ("Howdy", vec!["Howdy", "▁Howdy"]), // Added Howdy token for debugging
        ("Hi", vec!["Hi", "▁Hi"]),    // Added Hi token for debugging
    ];

    println!("\n🔍 Checking specific expected tokens:");
    for (_desc, variants) in &test_tokens {
        for variant in variants {
            if let Some(token_id) = tokenizer.token_to_id(variant) {
                let logit = logits[token_id as usize];
                let rank =
                    indexed_logits.iter().position(|(id, _)| *id == token_id as usize).unwrap_or(0);
                println!("   {}: id={}, logit={:.6}, rank={}", variant, token_id, logit, rank + 1);
            }
        }
    }

    let max_token = indexed_logits[0].0;
    let _max_logit = indexed_logits[0].1;
    let token_text = tokenizer.id_to_token(max_token as u32);
    let token_str = token_text.unwrap_or("<unknown>");

    // Compare with actual Ollama behavior
    println!("\n6️⃣ Analysis");
    println!("------------");
    println!("Our greedy output: '{}'", token_str);
    println!("Expected behavior: Should be deterministic with greedy sampling");

    // Check if we're getting reasonable tokens
    if token_str == "global" {
        println!("⚠️  Getting 'global' - this might indicate:");
        println!("   - Model weights loading issue");
        println!("   - Architecture mismatch");
        println!("   - Quantization/dequantization bug");
    } else {
        println!("✅ Getting reasonable token: '{}'", token_str);
    }

    Ok(())
}
