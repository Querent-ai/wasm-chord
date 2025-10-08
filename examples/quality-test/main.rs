use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 WASM-Chord Quality Test");
    println!("==========================");

    // Load model
    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";
    println!("📂 Loading model: {}", model_path);

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config from GGUF")?;
    let config: TransformerConfig = config_data.into();
    println!("✅ Model loaded: {} layers, {} vocab", config.num_layers, config.vocab_size);

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Create model and load weights
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

    // Reopen file for loading
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model weights loaded");

    // Test prompts with predictable completions
    let test_cases = vec![
        ("Q: What is the capital of France?\nA:", "Paris"),
        ("2 + 2 =", "4"),
        ("The first president of the United States was", "George Washington"),
        ("The color of the sky is", "blue"),
        ("Hello", "there"), // Simple greeting
    ];

    // Test with greedy (deterministic) sampling
    let greedy_config = GenerationConfig {
        temperature: 0.0, // Always pick highest probability
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        max_tokens: 5,
    };

    println!("\n🎯 Testing with Greedy Sampling (Temperature=0.0)");
    println!("==================================================");

    for (prompt, expected) in &test_cases {
        println!("\n📝 Prompt: \"{}\"", prompt);
        println!("🎯 Expected: \"{}\"", expected);

        // Tokenize prompt
        let tokens = tokenizer.encode(prompt, false)?;
        println!("🔢 Tokens: {:?}", tokens);

        // Generate with greedy sampling
        let result = model.generate(prompt, &tokenizer, &greedy_config)?;
        println!("🤖 Generated: \"{}\"", result);

        // Check if result contains expected word
        let contains_expected = result.to_lowercase().contains(&expected.to_lowercase());
        if contains_expected {
            println!("✅ SUCCESS: Contains expected word");
        } else {
            println!("❌ MISMATCH: Does not contain expected word");
        }

        // Show top 5 tokens for analysis
        println!("🔍 Analysis:");
        let forward_result = model.forward(&tokens, 0)?;
        let mut indexed: Vec<(usize, f32)> =
            forward_result.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
            let token_text = tokenizer.id_to_token(*idx as u32).unwrap_or("<unknown>");
            println!("   {}: {} (id: {}, logit: {:.6})", i + 1, token_text, idx, val);
        }
    }

    // Test with different sampling strategies
    println!("\n🎲 Testing Different Sampling Strategies");
    println!("========================================");

    let test_prompt = "The capital of France is";
    println!("\n📝 Test prompt: \"{}\"", test_prompt);

    let strategies = vec![
        (
            "Greedy",
            GenerationConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                max_tokens: 3,
            },
        ),
        (
            "Conservative",
            GenerationConfig {
                temperature: 0.3,
                top_p: 0.9,
                top_k: 40,
                repetition_penalty: 1.1,
                max_tokens: 3,
            },
        ),
        (
            "Creative",
            GenerationConfig {
                temperature: 0.8,
                top_p: 0.95,
                top_k: 50,
                repetition_penalty: 1.05,
                max_tokens: 3,
            },
        ),
    ];

    for (name, config) in strategies {
        println!("\n🎯 {} Sampling:", name);
        let result = model.generate(test_prompt, &tokenizer, &config)?;
        println!("   Result: \"{}\"", result);
    }

    // Test token-by-token analysis
    println!("\n🔬 Token-by-Token Analysis");
    println!("==========================");

    let analysis_prompt = "Hello";
    println!("\n📝 Analyzing prompt: \"{}\"", analysis_prompt);

    let tokens = tokenizer.encode(analysis_prompt, false)?;
    println!("🔢 Input tokens: {:?}", tokens);

    // Get logits for the last token
    let logits = model.forward(&tokens, tokens.len() - 1)?;

    // Find "Yes" token
    let yes_token = tokenizer.encode("Yes", false)?;
    if let Some(&yes_id) = yes_token.first() {
        println!("🔍 'Yes' token analysis:");
        println!("   Token ID: {}", yes_id);
        println!("   Logit: {:.6}", logits[yes_id as usize]);

        // Find rank
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let rank = indexed.iter().position(|(idx, _)| *idx == yes_id as usize).unwrap_or(0) + 1;
        println!("   Rank: {}", rank);

        // Show top 10 for context
        println!("   Top 10 tokens:");
        for (i, (idx, val)) in indexed.iter().take(10).enumerate() {
            let token_text = tokenizer.id_to_token(*idx as u32).unwrap_or("<unknown>");
            println!("     {}: {} (id: {}, logit: {:.6})", i + 1, token_text, idx, val);
        }
    }

    println!("\n✅ Quality test completed!");
    Ok(())
}
