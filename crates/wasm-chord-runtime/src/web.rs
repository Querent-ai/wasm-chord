//! WebAssembly bindings for browser usage
#![cfg(target_arch = "wasm32")]

use crate::{
    ChatMessage, ChatRole, ChatTemplate, GenerationConfig, Model as RustModel, TransformerConfig,
};
use std::io::Cursor;
use wasm_bindgen::prelude::*;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};

/// JavaScript-compatible Model wrapper
#[wasm_bindgen]
pub struct WasmModel {
    model: RustModel,
    tokenizer: Tokenizer,
    config: GenerationConfig,
}

#[wasm_bindgen]
impl WasmModel {
    /// Load model from GGUF bytes
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8]) -> Result<WasmModel, JsValue> {
        // Parse GGUF
        let cursor = Cursor::new(gguf_bytes);
        let mut parser = GGUFParser::new(cursor);
        let meta = parser
            .parse_header()
            .map_err(|e| JsValue::from_str(&format!("Failed to parse GGUF: {}", e)))?;

        let config_data =
            parser.extract_config().ok_or_else(|| JsValue::from_str("Failed to extract config"))?;
        let config: TransformerConfig = config_data.into();

        // Load tokenizer
        let tokenizer = Tokenizer::from_gguf(&meta)
            .map_err(|e| JsValue::from_str(&format!("Failed to load tokenizer: {}", e)))?;

        // Load weights
        let mut model = RustModel::new(config.clone());
        let data_offset = parser
            .tensor_data_offset()
            .map_err(|e| JsValue::from_str(&format!("Failed to get data offset: {}", e)))?;
        let mut tensor_loader = TensorLoader::new(data_offset);

        for tensor_desc in meta.tensors.iter() {
            tensor_loader.register_tensor(
                tensor_desc.name.clone(),
                tensor_desc.clone(),
                tensor_desc.offset,
            );
        }

        // Re-parse for loading
        let cursor = Cursor::new(gguf_bytes);
        let mut parser = GGUFParser::new(cursor);
        parser
            .parse_header()
            .map_err(|e| JsValue::from_str(&format!("Failed to re-parse: {}", e)))?;

        model
            .load_from_gguf(&mut tensor_loader, &mut parser)
            .map_err(|e| JsValue::from_str(&format!("Failed to load weights: {}", e)))?;

        Ok(WasmModel { model, tokenizer, config: GenerationConfig::default() })
    }

    /// Set generation configuration
    pub fn set_config(
        &mut self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
    ) {
        self.config = GenerationConfig {
            max_tokens,
            temperature,
            top_p,
            top_k: top_k as usize,
            repetition_penalty,
        };
    }

    /// Generate text (blocking)
    pub fn generate(&mut self, prompt: &str) -> Result<String, JsValue> {
        self.model
            .generate(prompt, &self.tokenizer, &self.config)
            .map_err(|e| JsValue::from_str(&format!("Generation failed: {}", e)))
    }

    /// Generate with streaming callback
    /// callback: function(token_text: string) -> boolean (continue?)
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        callback: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let this = JsValue::null();

        self.model
            .generate_stream(prompt, &self.tokenizer, &self.config, |_token_id, token_text| {
                let token_js = JsValue::from_str(token_text);
                if let Ok(result) = callback.call1(&this, &token_js) {
                    result.as_bool().unwrap_or(true)
                } else {
                    false
                }
            })
            .map_err(|e| JsValue::from_str(&format!("Streaming generation failed: {}", e)))
    }

    /// Generate with async iterator (returns AsyncTokenStream)
    /// Usage: for await (const token of model.generate_async(prompt)) { ... }
    pub fn generate_async(&mut self, prompt: String) -> AsyncTokenStream {
        AsyncTokenStream::new(prompt, self.config.clone())
    }

    /// Initialize GPU backend (if available)
    #[cfg(feature = "gpu")]
    pub fn init_gpu(&mut self) -> Result<(), JsValue> {
        self.model
            .init_gpu()
            .map_err(|e| JsValue::from_str(&format!("GPU initialization failed: {}", e)))
    }
}

/// Async token stream for streaming generation
#[wasm_bindgen]
pub struct AsyncTokenStream {
    prompt: String,
    config: GenerationConfig,
    tokens: Vec<String>,
    current_index: usize,
}

#[wasm_bindgen]
impl AsyncTokenStream {
    /// Create a new async token stream
    fn new(prompt: String, config: GenerationConfig) -> Self {
        Self { prompt, config, tokens: Vec::new(), current_index: 0 }
    }

    /// Get next token (async iterator protocol)
    /// Returns {value: string, done: boolean}
    pub async fn next(&mut self) -> Result<JsValue, JsValue> {
        // This is a simplified implementation
        // In a real implementation, we'd need to:
        // 1. Keep model state between calls
        // 2. Generate one token at a time
        // 3. Return each token as it's generated

        if self.current_index < self.tokens.len() {
            let token = self.tokens[self.current_index].clone();
            self.current_index += 1;

            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"value".into(), &token.into())?;
            js_sys::Reflect::set(&result, &"done".into(), &false.into())?;
            Ok(result.into())
        } else {
            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"value".into(), &JsValue::undefined())?;
            js_sys::Reflect::set(&result, &"done".into(), &true.into())?;
            Ok(result.into())
        }
    }

    /// Symbol.asyncIterator support
    #[wasm_bindgen(js_name = "Symbol.asyncIterator")]
    pub fn symbol_async_iterator(&self) -> AsyncTokenStream {
        AsyncTokenStream {
            prompt: self.prompt.clone(),
            config: self.config.clone(),
            tokens: self.tokens.clone(),
            current_index: 0,
        }
    }
}

/// Format chat messages with template
#[wasm_bindgen]
pub fn format_chat(
    system: Option<String>,
    user: String,
    template_type: &str,
) -> Result<String, JsValue> {
    let mut messages = Vec::new();

    if let Some(sys) = system {
        messages.push(ChatMessage { role: ChatRole::System, content: sys });
    }

    messages.push(ChatMessage { role: ChatRole::User, content: user });

    let template = match template_type {
        "chatml" => ChatTemplate::ChatML,
        "llama2" => ChatTemplate::Llama2,
        "alpaca" => ChatTemplate::Alpaca,
        _ => ChatTemplate::ChatML,
    };

    template
        .format(&messages)
        .map_err(|e| JsValue::from_str(&format!("Template formatting failed: {}", e)))
}

/// Get library version
#[wasm_bindgen]
pub fn version() -> String {
    crate::VERSION.to_string()
}
