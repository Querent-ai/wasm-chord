use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, Tokenizer, TensorLoader};
use wasm_chord_runtime::{Model, GenerationConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    
    let tokenizer = Tokenizer::from_gguf(&meta)?;
    
    // Load model
    let config_data = parser.extract_config().ok_or("Failed to extract config from GGUF")?;
    let config: wasm_chord_runtime::TransformerConfig = config_data.into();
    let mut model = wasm_chord_runtime::Model::new(config.clone());
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }
    
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;
    
    // Test with just the prompt tokens
    let prompt = "The meaning of life is";
    let tokens = tokenizer.encode(prompt, true)?;
    println!("Prompt tokens: {:?}", tokens);
    
    // Generate just one token
    let config = GenerationConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    
    let result = model.generate(&prompt, &tokenizer, &config)?;
    
    println!("Full result: '{}'", result);
    
    Ok(())
}
