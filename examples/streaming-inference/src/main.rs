/// Streaming inference example demonstrating real-time text generation
use std::io::Write;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::streaming::StreamingInference;
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Streaming Inference Demo");
    println!("==========================");

    let model_path = "/home/puneet/wasm-chord/models/tinyllama-1.1b.Q4_K_M.gguf";
    println!("📂 Loading model: {}", model_path);

    let file = std::fs::File::open(model_path)?;
    let reader = std::io::BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    println!("✅ Config: {} layers, {} vocab", config.num_layers, config.vocab_size);

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    println!("✅ Tokenizer: {} tokens", tokenizer.vocab_size());

    let mut model = Model::new(config.clone());
    println!("📦 Loading weights...");
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let file = std::fs::File::open(model_path)?;
    let reader = std::io::BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Weights loaded");

    let generation_config = GenerationConfig {
        max_tokens: 30,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    // Create streaming inference handler
    let mut streaming = StreamingInference::new(
        model,
        tokenizer,
        generation_config,
        Some(128), // Max sequence length
    );

    // Test prompts for streaming generation
    let test_prompts =
        vec!["The future of AI is", "Once upon a time", "The secret to happiness is"];

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n---");
        println!("📝 Test #{} - Streaming Generation", i + 1);
        println!("Prompt: \"{}\"", prompt);
        print!("Output: \"{}\" ", prompt);
        std::io::stdout().flush()?;

        // Start streaming
        streaming.start_streaming(prompt)?;

        // Generate tokens one by one in streaming fashion
        let max_tokens = 15;
        for _ in 0..max_tokens {
            if let Some(token_text) = streaming.generate_next_token()? {
                print!("{}", token_text);
                std::io::stdout().flush()?;
            } else {
                break; // No more tokens
            }
        }
        println!("\"");

        // Reset for next prompt
        streaming.reset();
    }

    println!("\n✅ Streaming inference demo completed!");
    Ok(())
}
