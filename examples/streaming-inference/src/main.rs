/// Streaming inference example demonstrating real-time text generation
use std::fs::File;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Streaming Inference Demo");
    println!("==========================");

    let model_path = "/home/puneet/wasm-chord/models/tinyllama-1.1b.Q4_K_M.gguf";
    println!("📂 Loading model: {}", model_path);

    let mut file = std::fs::File::open(model_path)?;
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
        max_tokens: 20,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    // Test prompts with streaming-like behavior
    let test_prompts =
        vec!["The future of AI is", "Once upon a time", "The secret to happiness is"];

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n---\n📝 Test #{} Prompt: {:?}", i + 1, prompt);

        // Generate text
        let output = model.generate(prompt, &tokenizer, &generation_config)?;
        println!("🎯 Output: {:?}", output);
    }

    println!("\n✅ Streaming inference demo completed!");
    Ok(())
}
