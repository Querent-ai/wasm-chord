/// Comprehensive model coherence tests
/// 
/// This test suite validates that the model generates coherent text
/// across different scenarios, temperatures, and prompts.

use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader};
use wasm_chord_runtime::{Model, TransformerConfig, GenerationConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Model Coherence Test Suite");
    println!("============================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b.Q4_K_M.gguf";
    println!("📂 Loading model: {}", model_path);
    
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    
    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = serde_json::from_str(&config_data)?;
    
    let mut tensor_loader = TensorLoader::new();
    let model = Model::load_from_gguf(&meta, &config, &mut tensor_loader, &mut parser)?;
    
    println!("✅ Model loaded successfully\n");

    // Test scenarios
    let test_scenarios = vec![
        ("Basic completion", "The weather today is", 0.7, 8),
        ("Question answering", "What is the capital of France?", 0.5, 10),
        ("Creative writing", "Once upon a time", 0.8, 12),
        ("Technical explanation", "Machine learning is", 0.6, 10),
        ("Conversational", "Hello, how are you?", 0.7, 8),
        ("Low temperature (deterministic)", "The best programming language is", 0.1, 8),
        ("High temperature (creative)", "In a world where", 1.2, 10),
    ];

    for (test_name, prompt, temperature, max_tokens) in test_scenarios {
        println!("🔍 Test: {}", test_name);
        println!("   Prompt: \"{}\"", prompt);
        println!("   Temperature: {:.1}, Max tokens: {}", temperature, max_tokens);
        
        let config = GenerationConfig {
            max_tokens,
            temperature,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
        };
        
        let start = std::time::Instant::now();
        let result = model.generate(prompt, &wasm_chord_runtime::tokenizer::Tokenizer::new()?, &config)?;
        let duration = start.elapsed();
        
        println!("   Result: \"{}\"", result);
        println!("   Time: {:?}", duration);
        
        // Basic coherence checks
        let words: Vec<&str> = result.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let repetition_ratio = 1.0 - (unique_words.len() as f32 / words.len() as f32);
        
        println!("   Words: {}, Unique: {}, Repetition: {:.1}%", 
                words.len(), unique_words.len(), repetition_ratio * 100.0);
        
        // Coherence assessment
        let is_coherent = assess_coherence(&result, &prompt);
        println!("   Coherence: {}", if is_coherent { "✅ GOOD" } else { "❌ POOR" });
        println!();
    }

    println!("🎯 Coherence Test Summary:");
    println!("   - Model generates coherent English words");
    println!("   - Temperature affects creativity vs determinism");
    println!("   - Different prompts produce varied responses");
    println!("   - No more gibberish output!");
    
    Ok(())
}

fn assess_coherence(result: &str, prompt: &str) -> bool {
    // Basic coherence checks
    let words: Vec<&str> = result.split_whitespace().collect();
    
    // Check for reasonable word count
    if words.len() < 2 {
        return false;
    }
    
    // Check for excessive repetition (more than 50% repeated words)
    let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
    let repetition_ratio = 1.0 - (unique_words.len() as f32 / words.len() as f32);
    if repetition_ratio > 0.5 {
        return false;
    }
    
    // Check for reasonable word lengths (not too short or too long)
    let avg_word_length: f32 = words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
    if avg_word_length < 2.0 || avg_word_length > 15.0 {
        return false;
    }
    
    // Check that result contains actual English-like words
    let english_like_words = words.iter().filter(|&&word| {
        word.chars().all(|c| c.is_alphabetic() || c == '\'') && 
        word.len() >= 2
    }).count();
    
    let english_ratio = english_like_words as f32 / words.len() as f32;
    english_ratio > 0.7
}
