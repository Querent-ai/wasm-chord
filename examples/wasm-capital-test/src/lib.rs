//! WASM test for capital of France inference
//!
//! This example tests that the WASM build can correctly answer
//! "What is the capital of France?" with "Paris"

use wasm_bindgen::prelude::*;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Test function that will be called from JavaScript
/// Returns true if the model correctly identifies Paris
#[wasm_bindgen]
pub async fn test_capital_inference(model_bytes: &[u8]) -> Result<JsValue, JsValue> {
    use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
    use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};
    use std::io::Cursor;

    // Parse GGUF model
    let cursor = Cursor::new(model_bytes);
    let mut parser = GGUFParser::new(cursor);
    let meta = parser.parse_header()
        .map_err(|e| JsValue::from_str(&format!("Failed to parse header: {}", e)))?;

    // Extract config
    let config_data = parser.extract_config()
        .ok_or_else(|| JsValue::from_str("Failed to extract config"))?;
    let config: TransformerConfig = config_data.into();

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)
        .map_err(|e| JsValue::from_str(&format!("Failed to load tokenizer: {}", e)))?;

    // Create model
    let mut model = Model::new(config);

    // Load weights
    let data_offset = parser.tensor_data_offset()
        .map_err(|e| JsValue::from_str(&format!("Failed to get tensor offset: {}", e)))?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor in meta.tensors.iter() {
        tensor_loader.register_tensor(tensor.name.clone(), tensor.clone(), tensor.offset);
    }

    // Reopen for loading
    let cursor = Cursor::new(model_bytes);
    let mut parser = GGUFParser::new(cursor);
    parser.parse_header()
        .map_err(|e| JsValue::from_str(&format!("Failed to reparse header: {}", e)))?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)
        .map_err(|e| JsValue::from_str(&format!("Failed to load weights: {}", e)))?;

    // Test prompt
    let prompt = "What is the capital of France?";

    let gen_config = GenerationConfig {
        max_tokens: 20,
        temperature: 0.0, // Greedy
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };

    // Generate response
    let response = model.generate(prompt, &tokenizer, &gen_config)
        .map_err(|e| JsValue::from_str(&format!("Generation failed: {}", e)))?;

    // Check if response contains "Paris"
    let success = response.to_lowercase().contains("paris");

    // Return result as JSON
    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"prompt".into(), &JsValue::from_str(prompt))?;
    js_sys::Reflect::set(&result, &"response".into(), &JsValue::from_str(&response))?;
    js_sys::Reflect::set(&result, &"success".into(), &JsValue::from_bool(success))?;

    Ok(result.into())
}

/// Synchronous version for simpler testing
#[wasm_bindgen]
pub fn get_test_info() -> String {
    "WASM Capital Test - Tests inference for 'What is the capital of France?'".to_string()
}
