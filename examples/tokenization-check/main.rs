/// Tokenization Comparison Tool
/// This loads the GGUF tokenizer and checks if "Hello" maps to the same token ID as Ollama
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Tokenization Comparison Tool");
    println!("===============================\n");

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    // Load GGUF and extract tokenizer
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    println!("✅ GGUF loaded successfully");
    println!("   Architecture: {:?}", meta.architecture);
    println!("   Tensor count: {}", meta.tensor_count);

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Test various prompts
    let test_prompts = vec![
        "Hello",
        "Hello ",
        "hello",
        "The",
        "Once",
        "def",
        "import",
        "function",
        "class",
        "if",
        "for",
        "while",
        "return",
        "print",
        "console.log",
    ];

    println!("\n📝 Tokenization Results:");
    println!("========================");

    for prompt in test_prompts {
        match tokenizer.encode(prompt, false) {
            Ok(tokens) => {
                println!("'{:12}' → {:?}", prompt, tokens);

                // Also show what these tokens decode to
                let decoded: Vec<String> = tokens
                    .iter()
                    .map(|&token_id| {
                        tokenizer.id_to_token(token_id).unwrap_or("<unknown>").to_string()
                    })
                    .collect();
                println!("              → {:?}", decoded);
            }
            Err(e) => {
                println!("'{:12}' → ERROR: {}", prompt, e);
            }
        }
    }

    // Test specific "Hello" case
    println!("\n🎯 Detailed 'Hello' Analysis:");
    println!("=============================");

    let hello_tokens = tokenizer.encode("Hello", false)?;
    println!("'Hello' tokens: {:?}", hello_tokens);

    if let Some(&token_id) = hello_tokens.first() {
        let token_text = tokenizer.id_to_token(token_id);
        println!("First token ID: {}", token_id);
        println!("First token text: {:?}", token_text);

        // Check if this matches what we expect
        println!("\n🔍 Comparison:");
        println!("Our tokenization: 'Hello' → [{}] → {:?}", token_id, token_text);
        println!("Expected (Ollama): 'Hello' → [15043] (from our previous test)");

        if token_id == 15043 {
            println!("✅ Tokenization MATCHES! The issue is elsewhere.");
        } else {
            println!("❌ Tokenization MISMATCH! This explains the divergence.");
            println!("   Our tokenizer produces different token IDs than Ollama.");
        }
    }

    // Test vocabulary size and some key tokens
    println!("\n📚 Vocabulary Analysis:");
    println!("=======================");
    println!("Total vocabulary size: {}", tokenizer.vocab_size());

    // Check some common tokens
    let common_tokens = vec![
        "Hello", "hello", "The", "the", "A", "a", "I", "i", "You", "you", "Yes", "yes", "No", "no",
        "Good", "good", "Bad", "bad", "番", "Yes", "No", "Hello", "World", "world",
    ];

    println!("\nCommon token mappings:");
    for token in common_tokens {
        if let Ok(tokens) = tokenizer.encode(token, false) {
            if let Some(&token_id) = tokens.first() {
                let decoded = tokenizer.id_to_token(token_id);
                println!("'{:8}' → [{}] → {:?}", token, token_id, decoded);
            }
        }
    }

    Ok(())
}
