//! Example: End-to-end inference with GGUF model
//!
//! This example demonstrates the complete inference pipeline.

use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::formats::gguf::GGUFParser;
use wasm_chord_core::tensor_loader::TensorLoader;
use wasm_chord_core::tokenizer::{SpecialTokens, Tokenizer};
use wasm_chord_runtime::{GenOptions, InferenceSession, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <prompt>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = &args[2];

    println!("Loading model from: {}", model_path);

    // Parse GGUF file
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let _meta = parser.parse_header()?;

    println!("GGUF file parsed successfully");

    // Create model config (TinyLlama defaults)
    let config = TransformerConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 4,
        intermediate_size: 5632,
        max_seq_len: 2048,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        attention_backend: wasm_chord_runtime::attention::AttentionBackend::Auto,
    };

    println!("Model config: {} layers, {} hidden size", config.num_layers, config.hidden_size);

    // Create tokenizer
    let tokenizer = {
        use std::collections::HashMap;

        let mut vocab = HashMap::new();
        for i in 0..config.vocab_size {
            vocab.insert(format!("token_{}", i), i as u32);
        }

        Tokenizer::new(vocab, Vec::new(), SpecialTokens::default())
    };

    println!("Tokenizer created (vocab size: {})", tokenizer.vocab_size());

    // Create model and load weights
    let mut model = Model::new(config);
    let mut tensor_loader = TensorLoader::new(parser.data_offset());

    for (name, desc, offset) in parser.tensor_info() {
        tensor_loader.register_tensor(name.clone(), desc.clone(), offset);
    }

    println!("Loading weights from GGUF...");
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("Weights loaded successfully");

    // Encode prompt
    let prompt_tokens = tokenizer.encode(prompt, true)?;
    println!("Prompt: \"{}\" ({} tokens)", prompt, prompt_tokens.len());

    // Create inference session
    let mut session = InferenceSession::new(
        0,
        prompt_tokens,
        GenOptions {
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            seed: 0,
            stop_token_count: 0,
            stop_tokens_ptr: 0,
        },
    );

    session.set_stop_tokens(vec![tokenizer.special_tokens().eos_token_id]);

    // Generate tokens
    println!("\nGenerated text:");
    print!("{}", prompt);

    while let Some(token_id) = session.next_token_with_model(&mut model)? {
        let text = tokenizer.decode(&[token_id], true)?;
        print!("{}", text);
        use std::io::Write;
        std::io::stdout().flush()?;
    }

    println!("\n\nGeneration complete ({} tokens)", session.tokens_generated());

    Ok(())
}
