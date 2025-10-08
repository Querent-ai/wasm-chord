/// Quick test to verify the embedding fix works
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Quick Embedding Fix Test");
    println!("===========================\n");

    // Load model
    let model_path = "models/tinyllama-1.1b.Q4_0.gguf"; // Using fresh model from Oct 7
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    // Load model
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let mut model = Model::new(config.clone());
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    println!("✅ Model loaded successfully");

    // Test forward pass with BOS token
    let test_tokens = vec![1u32]; // BOS token
    println!("\n🧪 Testing forward pass with BOS token (ID=1):");

    // Use generation instead of direct forward to see tokenization debugging
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    let config = wasm_chord_runtime::GenerationConfig {
        max_tokens: 2,    // Generate exactly 2 tokens
        temperature: 0.7, // Add randomness to reduce repetition
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
    };

    match model.generate("Hello", &tokenizer, &config) {
        Ok(result) => {
            println!("Generated: {}", result);
        }
        Err(e) => {
            println!("❌ Generation failed: {}", e);
        }
    }

    Ok(())
}
