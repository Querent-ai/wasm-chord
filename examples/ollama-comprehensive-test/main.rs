/// Comprehensive Ollama Comparison Test
/// Tests multiple prompts to verify our implementation matches Ollama's behavior

use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Comprehensive Ollama Comparison Test");
    println!("==========================================\n");

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
    println!("   Config: vocab_size={}, hidden_size={}, num_layers={}", 
             config.vocab_size, config.hidden_size, config.num_layers);
    
    // Test cases with expected Ollama behavior
    let test_cases = vec![
        ("Hello", "Yes"),
        ("The", "cat"),
        ("Once", "upon"),
        ("def", " "),
        ("import", " "),
        ("function", " "),
        ("class", " "),
        ("if", " "),
        ("for", " "),
        ("while", " "),
        ("return", " "),
        ("print", " "),
        ("console.log", "("),
        ("Hello world", "!"),
        ("The quick", "brown"),
        ("Once upon a", "time"),
    ];
    
    println!("\n🔍 Testing Multiple Prompts:");
    println!("============================");
    
    let mut matches = 0;
    let mut total_tests = test_cases.len();
    
    for (prompt, expected_start) in test_cases {
        // Tokenize prompt
        let tokens = tokenizer.encode(prompt, false)?;
        
        // Generate first token
        let logits = model.forward(&tokens, 0)?;
        
        // Find top token
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
        
        // Check if prediction starts with expected text
        let is_match = predicted_text.starts_with(expected_start);
        
        println!("'{:12}' → '{:8}' (expected: '{:8}') {}", 
                 prompt, predicted_text, expected_start, 
                 if is_match { "✅" } else { "❌" });
        
        if is_match {
            matches += 1;
        }
        
        // Show top 3 predictions for debugging
        let mut sorted_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        sorted_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        print!("   Top 3: ");
        for i in 0..3 {
            let (token_id, logit) = sorted_logits[i];
            let token_text = tokenizer.id_to_token(token_id as u32).unwrap_or("<unknown>");
            print!("'{}'({:.2}) ", token_text, logit);
        }
        println!();
    }
    
    println!("\n📊 Test Results:");
    println!("================");
    println!("Matches: {}/{} ({:.1}%)", matches, total_tests, (matches as f32 / total_tests as f32) * 100.0);
    
    if matches >= total_tests * 8 / 10 {
        println!("🎉 EXCELLENT! Our implementation closely matches Ollama!");
    } else if matches >= total_tests * 6 / 10 {
        println!("✅ GOOD! Our implementation is mostly correct with minor differences.");
    } else if matches >= total_tests * 4 / 10 {
        println!("⚠️  PARTIAL! Some issues remain but we're on the right track.");
    } else {
        println!("❌ NEEDS WORK! Significant differences from Ollama.");
    }
    
    // Test specific known cases
    println!("\n🎯 Specific Known Cases:");
    println!("========================");
    
    let known_cases = vec![
        ("Hello", "Yes", "Ollama's known first token for 'Hello'"),
        ("The", "cat", "Common continuation for 'The'"),
        ("Once", "upon", "Common continuation for 'Once'"),
    ];
    
    for (prompt, expected, description) in known_cases {
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
        
        println!("'{:8}' → '{:8}' (expected: '{:8}') - {}", 
                 prompt, predicted_text, expected, description);
        
        if predicted_text == expected {
            println!("   ✅ PERFECT MATCH!");
        } else {
            println!("   ❌ Mismatch");
        }
    }
    
    // Test logit distribution
    println!("\n📈 Logit Distribution Analysis:");
    println!("===============================");
    
    let test_prompt = "Hello";
    let tokens = tokenizer.encode(test_prompt, false)?;
    let logits = model.forward(&tokens, 0)?;
    
    // Find statistics
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let mean_logit = logits.iter().sum::<f32>() / logits.len() as f32;
    
    // Count tokens above certain thresholds
    let above_0 = logits.iter().filter(|&&x| x > 0.0).count();
    let above_1 = logits.iter().filter(|&&x| x > 1.0).count();
    let above_5 = logits.iter().filter(|&&x| x > 5.0).count();
    
    println!("For prompt '{}':", test_prompt);
    println!("  Max logit: {:.6}", max_logit);
    println!("  Min logit: {:.6}", min_logit);
    println!("  Mean logit: {:.6}", mean_logit);
    println!("  Tokens > 0: {}", above_0);
    println!("  Tokens > 1: {}", above_1);
    println!("  Tokens > 5: {}", above_5);
    
    // Check if distribution looks reasonable
    if max_logit > 10.0 {
        println!("  ⚠️  Warning: Max logit is very high, may indicate scaling issues");
    }
    if above_5 > 100 {
        println!("  ⚠️  Warning: Too many high logits, may indicate overconfidence");
    }
    
    println!("\n🏁 Test Complete!");
    
    Ok(())
}
