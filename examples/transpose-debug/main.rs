/// Systematic transpose debugging
///
/// This test will help determine which weights need transposing by:
/// 1. Loading weights with and without transposes
/// 2. Testing matrix multiplication dimensions
/// 3. Comparing with expected GGML behavior
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, ModelMeta};
use wasm_chord_runtime::TransformerConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Transpose Debug Test");
    println!("=====================\n");

    let model_path = "models/tinyllama-1.1b.Q4_K_M.gguf";
    println!("📂 Loading model: {}", model_path);

    // Load model metadata
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();
    println!(
        "✅ Config: {} layers, {} vocab, {} hidden",
        config.num_layers, config.vocab_size, config.hidden_size
    );

    // First, let's see what tensors are available
    println!("\n📋 Available tensors:");
    for (i, tensor) in meta.tensors.iter().enumerate() {
        if i < 10 {
            // Show first 10 tensors
            println!("  {}: {:?}", tensor.name, tensor.shape.0);
        }
    }
    if meta.tensors.len() > 10 {
        println!("  ... and {} more tensors", meta.tensors.len() - 10);
    }

    // Test specific weight matrices (using actual tensor names)
    test_weight_matrix("token_embd.weight", &meta, &config)?;
    test_weight_matrix("output.weight", &meta, &config)?;
    test_weight_matrix("blk.0.attn_q.weight", &meta, &config)?;
    test_weight_matrix("blk.0.attn_k.weight", &meta, &config)?;
    test_weight_matrix("blk.0.attn_v.weight", &meta, &config)?;
    test_weight_matrix("blk.0.attn_output.weight", &meta, &config)?;

    Ok(())
}

fn test_weight_matrix(
    tensor_name: &str,
    meta: &ModelMeta,
    config: &TransformerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing tensor: {}", tensor_name);

    // Find tensor in metadata
    let tensor_desc = meta
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .ok_or_else(|| format!("Tensor {} not found", tensor_name))?;

    println!("  📐 GGUF shape: {:?}", tensor_desc.shape);
    println!("  📊 GGUF dtype: {:?}", tensor_desc.dtype);

    // Determine expected dimensions based on tensor name
    let (expected_rows, expected_cols) = match tensor_name {
        "token_embd.weight" => (config.vocab_size, config.hidden_size),
        "output.weight" => (config.vocab_size, config.hidden_size),
        "blk.0.attn_q.weight" => (config.hidden_size, config.hidden_size),
        "blk.0.attn_k.weight" => {
            let kv_dim = config.num_kv_heads * (config.hidden_size / config.num_heads);
            (kv_dim, config.hidden_size)
        }
        "blk.0.attn_v.weight" => {
            let kv_dim = config.num_kv_heads * (config.hidden_size / config.num_heads);
            (kv_dim, config.hidden_size)
        }
        "blk.0.attn_output.weight" => (config.hidden_size, config.hidden_size),
        _ => {
            println!("  ⚠️  Unknown tensor, skipping dimension analysis");
            return Ok(());
        }
    };

    println!("  🎯 Expected for matmul: [{} x {}]", expected_rows, expected_cols);

    // Check if GGUF shape matches expected
    let gguf_shape = &tensor_desc.shape.0;
    if gguf_shape.len() == 2 {
        let (gguf_rows, gguf_cols) = (gguf_shape[0], gguf_shape[1]);
        println!("  📋 GGUF actual: [{} x {}]", gguf_rows, gguf_cols);

        if gguf_rows == expected_rows && gguf_cols == expected_cols {
            println!("  ✅ GGUF shape matches expected - NO transpose needed");
        } else if gguf_rows == expected_cols && gguf_cols == expected_rows {
            println!("  🔄 GGUF shape is transposed - TRANSPOSE needed");
        } else {
            println!("  ❌ GGUF shape doesn't match expected dimensions!");
        }
    } else {
        println!("  ⚠️  Unexpected GGUF shape dimensions: {:?}", gguf_shape);
    }

    Ok(())
}
