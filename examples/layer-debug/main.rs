/// Layer-by-Layer Debug Tool
/// This tool traces activations through each transformer layer to find where amplification occurs
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Layer-by-Layer Debug Tool");
    println!("============================\n");

    let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

    // Load model
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    let mut model = Model::new(config.clone());

    // Load weights
    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;

    println!("✅ Model loaded successfully");
    println!(
        "   Config: vocab_size={}, hidden_size={}, num_layers={}",
        config.vocab_size, config.hidden_size, config.num_layers
    );

    // Test with simple token
    let test_token = "Hello";
    let tokens = tokenizer.encode(test_token, false)?;
    println!("\n🔍 Testing with token: '{}' -> {:?}", test_token, tokens);

    // Create a debug version of the model that logs intermediate activations
    let debug_model = DebugModel::new(model);

    // Run forward pass with debug logging
    let logits = debug_model.debug_forward(&tokens, 0)?;

    // Analyze final logits
    println!("\n🏁 Final Analysis:");
    println!("==================");

    let mut max_logit = f32::NEG_INFINITY;
    let mut max_token = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > max_logit {
            max_logit = logit;
            max_token = i;
        }
    }

    let predicted_token = tokenizer.id_to_token(max_token as u32);
    let predicted_text = predicted_token.unwrap_or("<unknown>");

    println!(
        "   Predicted token: '{}' (id: {}, logit: {:.6})",
        predicted_text, max_token, max_logit
    );

    // Check specific tokens
    let test_tokens = vec![
        ("Hello", tokens[0] as usize),
        ("Yes", 3869),
        ("No", 1939),
        ("番", 31982),
        ("rique", 29871),
    ];

    println!("\n📊 Final Token Logits:");
    println!("=====================");
    for (token_name, token_idx) in test_tokens {
        if token_idx < logits.len() {
            println!("   '{}' ({}): {:.6}", token_name, token_idx, logits[token_idx]);
        }
    }

    Ok(())
}

struct DebugModel {
    model: Model,
}

impl DebugModel {
    fn new(model: Model) -> Self {
        Self { model }
    }

    fn debug_forward(
        &self,
        token_ids: &[u32],
        position: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let seq_len = token_ids.len();
        let hidden_size = self.model.config.hidden_size;

        println!("\n🚀 Starting Debug Forward Pass");
        println!("===============================");
        println!("   Input tokens: {:?}", token_ids);
        println!("   Sequence length: {}", seq_len);
        println!("   Position: {}", position);
        println!("   Hidden size: {}", hidden_size);

        // 1. Token Embeddings
        println!("\n1️⃣ Token Embeddings");
        println!("===================");

        let mut hidden_states = vec![0.0; seq_len * hidden_size];
        for (seq_idx, &token_id) in token_ids.iter().enumerate() {
            let out_start = seq_idx * hidden_size;
            let emb_start = token_id as usize * hidden_size;

            for i in 0..hidden_size {
                hidden_states[out_start + i] = self.model.token_embeddings[emb_start + i];
            }
        }

        self.print_tensor_stats("Token Embeddings", &hidden_states);

        // 2. Process through each transformer layer
        for layer_idx in 0..self.model.config.num_layers {
            println!("\n🔄 Layer {} Processing", layer_idx);
            println!("========================");

            // Pre-attention RMSNorm
            println!("\n   📐 Pre-Attention RMSNorm");
            let norm_input = hidden_states.clone();
            hidden_states =
                self.debug_rms_norm(&hidden_states, &self.model.layers[layer_idx].attention_norm)?;
            self.print_tensor_stats("Pre-Attention RMSNorm Output", &hidden_states);
            self.print_tensor_diff("RMSNorm Change", &norm_input, &hidden_states);

            // Attention
            println!("\n   🎯 Attention");
            let attn_input = hidden_states.clone();
            let attn_output = self.debug_attention(&hidden_states, layer_idx, position)?;
            self.print_tensor_stats("Attention Output", &attn_output);
            self.print_tensor_diff("Attention Change", &attn_input, &attn_output);

            // Residual connection (attention)
            println!("\n   ➕ Residual Connection (Attention)");
            let residual_input = hidden_states.clone();
            for i in 0..hidden_states.len() {
                hidden_states[i] += attn_output[i];
            }
            self.print_tensor_stats("After Attention Residual", &hidden_states);
            self.print_tensor_diff("Attention Residual Change", &residual_input, &hidden_states);

            // Pre-FFN RMSNorm
            println!("\n   📐 Pre-FFN RMSNorm");
            let norm_input = hidden_states.clone();
            hidden_states =
                self.debug_rms_norm(&hidden_states, &self.model.layers[layer_idx].ffn_norm)?;
            self.print_tensor_stats("Pre-FFN RMSNorm Output", &hidden_states);
            self.print_tensor_diff("FFN RMSNorm Change", &norm_input, &hidden_states);

            // FFN
            println!("\n   🧠 Feed-Forward Network");
            let ffn_input = hidden_states.clone();
            let ffn_output = self.debug_ffn(&hidden_states, layer_idx)?;
            self.print_tensor_stats("FFN Output", &ffn_output);
            self.print_tensor_diff("FFN Change", &ffn_input, &ffn_output);

            // Residual connection (FFN)
            println!("\n   ➕ Residual Connection (FFN)");
            let residual_input = hidden_states.clone();
            for i in 0..hidden_states.len() {
                hidden_states[i] += ffn_output[i];
            }
            self.print_tensor_stats("After FFN Residual", &hidden_states);
            self.print_tensor_diff("FFN Residual Change", &residual_input, &hidden_states);

            // Check for extreme values
            let max_abs = hidden_states.iter().map(|&x| x.abs()).fold(0.0, f32::max);
            if max_abs > 10.0 {
                println!("   ⚠️  WARNING: Extreme values detected! Max abs: {:.6}", max_abs);
            }
        }

        // 3. Final normalization
        println!("\n3️⃣ Final Normalization");
        println!("======================");
        let final_input = hidden_states.clone();
        hidden_states = self.debug_rms_norm(&hidden_states, &self.model.output_norm)?;
        self.print_tensor_stats("Final RMSNorm Output", &hidden_states);
        self.print_tensor_diff("Final RMSNorm Change", &final_input, &hidden_states);

        // 4. LM Head
        println!("\n4️⃣ Language Model Head");
        println!("======================");
        let _lm_input = hidden_states.clone();
        let logits = self.debug_lm_head(&hidden_states)?;
        self.print_tensor_stats("LM Head Output (Logits)", &logits);

        Ok(logits)
    }

    fn debug_rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let hidden_size = self.model.config.hidden_size;
        let mut output = vec![0.0; input.len()];

        for (offset, chunk) in input.chunks(hidden_size).enumerate() {
            let sum_sq: f32 = chunk.iter().map(|&x| x * x).sum();
            let mean = sum_sq / hidden_size as f32;
            let rms = (mean + self.model.config.rms_norm_eps).sqrt();

            for i in 0..hidden_size {
                output[offset * hidden_size + i] = (chunk[i] / rms) * weight[i];
            }
        }

        Ok(output)
    }

    fn debug_attention(
        &self,
        input: &[f32],
        _layer_idx: usize,
        _position: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // This is a simplified version - in reality we'd need to implement the full attention
        // For now, just return a placeholder that shows the structure
        let hidden_size = self.model.config.hidden_size;
        let _seq_len = input.len() / hidden_size;

        // Placeholder: just return input scaled by a small factor to show the flow
        let mut output = vec![0.0; input.len()];
        for i in 0..input.len() {
            output[i] = input[i] * 0.1; // Simulate attention scaling
        }

        Ok(output)
    }

    fn debug_ffn(
        &self,
        input: &[f32],
        _layer_idx: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // This is a simplified version - in reality we'd need to implement the full FFN
        // For now, just return a placeholder that shows the structure
        let hidden_size = self.model.config.hidden_size;
        let _seq_len = input.len() / hidden_size;

        // Placeholder: just return input scaled by a small factor to show the flow
        let mut output = vec![0.0; input.len()];
        for i in 0..input.len() {
            output[i] = input[i] * 0.1; // Simulate FFN scaling
        }

        Ok(output)
    }

    fn debug_lm_head(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let hidden_size = self.model.config.hidden_size;
        let vocab_size = self.model.config.vocab_size;
        let seq_len = input.len() / hidden_size;

        let mut logits = vec![0.0; seq_len * vocab_size];

        for seq_idx in 0..seq_len {
            for token_idx in 0..vocab_size {
                let mut sum = 0.0;
                for dim_idx in 0..hidden_size {
                    let input_idx = seq_idx * hidden_size + dim_idx;
                    let weight_idx = dim_idx * vocab_size + token_idx;
                    sum += input[input_idx] * self.model.lm_head[weight_idx];
                }
                logits[seq_idx * vocab_size + token_idx] = sum;
            }
        }

        Ok(logits)
    }

    fn print_tensor_stats(&self, name: &str, tensor: &[f32]) {
        let mean = tensor.iter().sum::<f32>() / tensor.len() as f32;
        let variance =
            tensor.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / tensor.len() as f32;
        let std = variance.sqrt();
        let min = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_abs = tensor.iter().map(|&x| x.abs()).fold(0.0, f32::max);

        println!(
            "     {}: mean={:.6}, std={:.6}, min={:.6}, max={:.6}, max_abs={:.6}",
            name, mean, std, min, max, max_abs
        );

        // Show first few values
        let preview_len = 5.min(tensor.len());
        let preview: Vec<String> =
            tensor[..preview_len].iter().map(|&x| format!("{:.6}", x)).collect();
        println!("     Preview: [{}]", preview.join(", "));
    }

    fn print_tensor_diff(&self, name: &str, before: &[f32], after: &[f32]) {
        if before.len() != after.len() {
            println!(
                "     {}: Length mismatch! before={}, after={}",
                name,
                before.len(),
                after.len()
            );
            return;
        }

        let mut max_diff: f32 = 0.0;
        let mut total_diff = 0.0;
        for i in 0..before.len() {
            let diff = (after[i] - before[i]).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff;
        }

        let avg_diff = total_diff / before.len() as f32;
        println!("     {}: max_diff={:.6}, avg_diff={:.6}", name, max_diff, avg_diff);
    }
}
