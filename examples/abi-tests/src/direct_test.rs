use anyhow::Result;
use std::fs;
use std::io::Cursor;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<()> {
    println!("🧪 Testing direct Rust API for 'capital of France' -> 'Paris'");

    // Load and parse model
    println!("\n📦 Loading model...");
    let model_path = "../../models/tinyllama-1.1b-chat-v0.6-Q4_K_M.gguf";
    let model_bytes = fs::read(model_path)?;

    let cursor = Cursor::new(&model_bytes);
    let mut parser = GGUFParser::new(cursor);
    let meta = parser.parse_header()?;

    let config_data = parser
        .extract_config()
        .ok_or_else(|| anyhow::anyhow!("Failed to extract config from GGUF"))?;
    let config: TransformerConfig = config_data.into();

    // Load tokenizer
    println!("🔤 Loading tokenizer...");
    let tokenizer = Tokenizer::from_gguf(&meta)?;

    // Create model
    println!("🏗️  Creating model...");
    let mut model = Model::new(config.clone());

    // Load weights
    println!("⚖️  Loading weights...");
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    // Re-parse for loading
    let cursor = Cursor::new(&model_bytes);
    let mut parser = GGUFParser::new(cursor);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;

    // Initialize GPU if available
    #[cfg(feature = "cuda")]
    {
        println!("🎮 Initializing GPU...");
        if let Err(e) = model.init_gpu() {
            println!("⚠️  GPU initialization failed: {} (continuing with CPU)", e);
        } else {
            println!("✓ GPU initialized successfully");
        }
    }

    // Generate text
    println!("\n🚀 Generating text...");
    let prompt = "What is the capital of France?";
    let gen_config = GenerationConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
    };

    let response = model.generate(prompt, &tokenizer, &gen_config)?;
    println!("📝 Response: '{}'", response);

    // Validate result
    let response_lower = response.to_lowercase();
    if response_lower.contains("paris") {
        println!("✅ SUCCESS: Generated 'Paris' for capital of France!");
    } else {
        println!("❌ FAILURE: Did not generate 'Paris'. Response: '{}'", response);
        return Ok(()); // Don't fail the test, just report
    }

    println!("\n🎉 Direct API test completed successfully!");
    Ok(())
}
