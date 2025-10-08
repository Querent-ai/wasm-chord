/// RMSNorm Debug Tool
/// This tool focuses specifically on RMSNorm implementations to find scaling issues
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 RMSNorm Debug Tool");
    println!("====================\n");

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
    println!("   RMS norm eps: {}", config.rms_norm_eps);

    // Test with simple token
    let test_token = "Hello";
    let tokens = tokenizer.encode(test_token, false)?;
    println!("\n🔍 Testing with token: '{}' -> {:?}", test_token, tokens);

    // Create a debug version of the model that logs RMSNorm operations
    let debug_model = RMSNormDebugModel::new(model);

    // Run forward pass with RMSNorm debug logging
    let logits = debug_model.debug_rmsnorm_forward(&tokens, 0)?;

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

    Ok(())
}

struct RMSNormDebugModel {
    model: Model,
}

impl RMSNormDebugModel {
    fn new(model: Model) -> Self {
        Self { model }
    }

    fn debug_rmsnorm_forward(
        &self,
        token_ids: &[u32],
        position: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let seq_len = token_ids.len();
        let hidden_size = self.model.config.hidden_size;

        println!("\n🚀 Starting RMSNorm Debug Forward Pass");
        println!("=====================================");
        println!("   Input tokens: {:?}", token_ids);
        println!("   Sequence length: {}", seq_len);
        println!("   Position: {}", position);
        println!("   Hidden size: {}", hidden_size);
        println!("   RMS norm eps: {}", self.model.config.rms_norm_eps);

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

        // 2. Process through each transformer layer with RMSNorm focus
        for layer_idx in 0..self.model.config.num_layers {
            println!("\n🔄 Layer {} RMSNorm Analysis", layer_idx);
            println!("=============================");

            // Pre-attention RMSNorm
            println!("\n   📐 Pre-Attention RMSNorm");
            let norm_input = hidden_states.clone();
            hidden_states = self.debug_rms_norm(
                &hidden_states,
                &self.model.layers[layer_idx].attention_norm,
                format!("Layer {} Pre-Attention", layer_idx),
            )?;

            // Attention (simplified)
            println!("\n   🎯 Attention (Simplified)");
            let attn_input = hidden_states.clone();
            let attn_output = self.simplified_attention(&hidden_states)?;

            // Residual connection (attention)
            println!("\n   ➕ Residual Connection (Attention)");
            let residual_input = hidden_states.clone();
            for i in 0..hidden_states.len() {
                hidden_states[i] += attn_output[i];
            }

            // Pre-FFN RMSNorm
            println!("\n   📐 Pre-FFN RMSNorm");
            let norm_input = hidden_states.clone();
            hidden_states = self.debug_rms_norm(
                &hidden_states,
                &self.model.layers[layer_idx].ffn_norm,
                format!("Layer {} Pre-FFN", layer_idx),
            )?;

            // FFN (simplified)
            println!("\n   🧠 Feed-Forward Network (Simplified)");
            let ffn_input = hidden_states.clone();
            let ffn_output = self.simplified_ffn(&hidden_states)?;

            // Residual connection (FFN)
            println!("\n   ➕ Residual Connection (FFN)");
            let residual_input = hidden_states.clone();
            for i in 0..hidden_states.len() {
                hidden_states[i] += ffn_output[i];
            }

            // Check for extreme values
            let max_abs = hidden_states.iter().map(|&x| x.abs()).fold(0.0, f32::max);
            if max_abs > 10.0 {
                println!("   ⚠️  WARNING: Extreme values detected! Max abs: {:.6}", max_abs);
            }
        }

        // 3. Final normalization
        println!("\n3️⃣ Final RMSNorm");
        println!("================");
        let final_input = hidden_states.clone();
        hidden_states = self.debug_rms_norm(
            &hidden_states,
            &self.model.output_norm,
            "Final Output".to_string(),
        )?;

        // 4. LM Head
        println!("\n4️⃣ Language Model Head");
        println!("======================");
        let logits = self.debug_lm_head(&hidden_states)?;

        Ok(logits)
    }

    fn debug_rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        name: String,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let hidden_size = self.model.config.hidden_size;
        let eps = self.model.config.rms_norm_eps;
        let mut output = vec![0.0; input.len()];

        println!("     Input stats:");
        self.print_tensor_stats("  Input", input);

        // Analyze weight statistics
        let weight_mean = weight.iter().sum::<f32>() / weight.len() as f32;
        let weight_std = {
            let variance = weight.iter().map(|&x| (x - weight_mean).powi(2)).sum::<f32>()
                / weight.len() as f32;
            variance.sqrt()
        };
        let weight_min = weight.iter().copied().fold(f32::INFINITY, f32::min);
        let weight_max = weight.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weight_max_abs = weight.iter().map(|&x| x.abs()).fold(0.0, f32::max);

        println!("     Weight stats:");
        println!(
            "       Mean: {:.6}, Std: {:.6}, Min: {:.6}, Max: {:.6}, Max Abs: {:.6}",
            weight_mean, weight_std, weight_min, weight_max, weight_max_abs
        );

        // Process each sequence position
        for (offset, chunk) in input.chunks(hidden_size).enumerate() {
            // Compute RMS
            let sum_sq: f32 = chunk.iter().map(|&x| x * x).sum();
            let mean = sum_sq / hidden_size as f32;
            let rms = (mean + eps).sqrt();

            // Debug RMS computation
            if offset == 0 {
                println!("     RMS computation (first sequence):");
                println!("       Sum of squares: {:.6}", sum_sq);
                println!("       Mean: {:.6}", mean);
                println!("       RMS: {:.6}", rms);
                println!("       Epsilon: {:.6}", eps);
                println!("       Mean + eps: {:.6}", mean + eps);
            }

            // Normalize and scale by weight
            for i in 0..hidden_size {
                let normalized = chunk[i] / rms;
                output[offset * hidden_size + i] = normalized * weight[i];
            }
        }

        println!("     Output stats:");
        self.print_tensor_stats("  Output", &output);

        // Check if RMSNorm is working correctly
        let input_max_abs = input.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let output_max_abs = output.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let amplification_factor =
            if input_max_abs > 0.0 { output_max_abs / input_max_abs } else { 0.0 };

        println!("     Amplification analysis:");
        println!("       Input max abs: {:.6}", input_max_abs);
        println!("       Output max abs: {:.6}", output_max_abs);
        println!("       Amplification factor: {:.6}", amplification_factor);

        if amplification_factor > 2.0 {
            println!("       ⚠️  WARNING: RMSNorm is amplifying values!");
        } else if amplification_factor < 0.5 {
            println!("       ⚠️  WARNING: RMSNorm is over-normalizing!");
        } else {
            println!("       ✅ RMSNorm scaling looks reasonable");
        }

        Ok(output)
    }

    fn simplified_attention(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Simplified attention - just scale down to simulate attention
        let mut output = vec![0.0; input.len()];
        for i in 0..input.len() {
            output[i] = input[i] * 0.1; // Simulate attention scaling
        }
        Ok(output)
    }

    fn simplified_ffn(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Simplified FFN - just scale down to simulate FFN
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
            "       {}: mean={:.6}, std={:.6}, min={:.6}, max={:.6}, max_abs={:.6}",
            name, mean, std, min, max, max_abs
        );

        // Show first few values
        let preview_len = 3.min(tensor.len());
        let preview: Vec<String> =
            tensor[..preview_len].iter().map(|&x| format!("{:.6}", x)).collect();
        println!("       Preview: [{}]", preview.join(", "));
    }
}
