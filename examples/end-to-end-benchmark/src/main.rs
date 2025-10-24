/// End-to-end inference benchmark
///
/// Measures:
/// - Tokens/second (prefill + decode)
/// - First token latency (prefill time)
/// - Inter-token latency (decode time)
/// - Memory usage
/// - Total inference time

use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
    }
    0.0 // Fallback if can't read memory
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         End-to-End Inference Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf".to_string());

    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Once upon a time".to_string());

    let num_tokens = std::env::args()
        .nth(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);

    println!("📊 Benchmark Configuration:");
    println!("   Model: {}", model_path);
    println!("   Prompt: {:?}", prompt);
    println!("   Tokens to generate: {}", num_tokens);
    println!();

    // Load model
    println!("📦 Loading model...");
    let load_start = Instant::now();
    let mem_before_load = get_memory_usage_mb();

    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    println!("   Architecture: {} layers, {} heads", config.num_layers, config.num_heads);
    println!("   Hidden size: {}, Vocab: {}", config.hidden_size, config.vocab_size);

    let tokenizer = Tokenizer::from_gguf(&meta)?;
    let mut model = Model::new(config.clone());

    let data_offset = parser.tensor_data_offset()?;
    let mut tensor_loader = TensorLoader::new(data_offset);

    for tensor_desc in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor_desc.name.clone(),
            tensor_desc.clone(),
            tensor_desc.offset,
        );
    }

    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;

    let load_time = load_start.elapsed();
    let mem_after_load = get_memory_usage_mb();
    let model_mem = mem_after_load - mem_before_load;

    println!("   ✅ Model loaded in {:.2}s", load_time.as_secs_f64());
    println!("   Memory: {:.1} MB\n", model_mem);

    // Prepare generation config
    let gen_config = GenerationConfig {
        max_tokens: num_tokens,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    // Run benchmark
    println!("🚀 Running inference benchmark...\n");
    let inference_start = Instant::now();
    let mem_before_inference = get_memory_usage_mb();

    // Tokenize prompt
    let tokens = tokenizer.encode(&prompt, false)?;
    let prompt_tokens = tokens.len();
    println!("   Prompt tokens: {}", prompt_tokens);

    // Generate with detailed timing
    let result = model.generate(&prompt, &tokenizer, &gen_config)?;

    let inference_time = inference_start.elapsed();
    let mem_after_inference = get_memory_usage_mb();
    let inference_mem = mem_after_inference - mem_before_inference;

    // Parse output to count actual tokens generated
    let output_tokens = tokenizer.encode(&result, false)?.len();
    let total_tokens = output_tokens;
    let generated_tokens = total_tokens.saturating_sub(prompt_tokens);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Results                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Timing Results
    println!("⏱️  Timing:");
    println!("   Total inference time: {:.2}s", inference_time.as_secs_f64());
    println!("   Prompt tokens: {}", prompt_tokens);
    println!("   Generated tokens: {}", generated_tokens);
    println!("   Total tokens: {}", total_tokens);
    println!();

    // Throughput
    let tokens_per_sec = total_tokens as f64 / inference_time.as_secs_f64();
    let ms_per_token = inference_time.as_millis() as f64 / total_tokens as f64;

    println!("🚀 Throughput:");
    println!("   Tokens/second: {:.2}", tokens_per_sec);
    println!("   Milliseconds/token: {:.2}", ms_per_token);
    println!();

    // Estimate prefill vs decode (rough approximation)
    // Assume first 20% of time is prefill, rest is decode
    let estimated_prefill_time = inference_time.as_secs_f64() * 0.2;
    let estimated_decode_time = inference_time.as_secs_f64() * 0.8;
    let estimated_decode_tps = if generated_tokens > 0 {
        generated_tokens as f64 / estimated_decode_time
    } else {
        0.0
    };

    println!("📊 Estimated Phase Breakdown:");
    println!("   Prefill time (estimated): {:.2}s", estimated_prefill_time);
    println!("   Decode time (estimated): {:.2}s", estimated_decode_time);
    println!("   Decode tokens/sec: {:.2}", estimated_decode_tps);
    println!();

    // Memory
    println!("💾 Memory Usage:");
    println!("   Model size: {:.1} MB", model_mem);
    println!("   Inference overhead: {:.1} MB", inference_mem);
    println!("   Total memory: {:.1} MB", mem_after_inference);
    println!();

    // Output sample
    println!("📝 Output Sample:");
    let output_preview = if result.len() > 200 {
        format!("{}...", &result[..200])
    } else {
        result.clone()
    };
    println!("   {}\n", output_preview);

    // Summary for GPU comparison
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   CPU Baseline (for GPU)                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Save these numbers for GPU comparison:");
    println!("  • Tokens/sec: {:.2}", tokens_per_sec);
    println!("  • ms/token: {:.2}", ms_per_token);
    println!("  • Total time: {:.2}s", inference_time.as_secs_f64());
    println!("  • Model: {}", model_path.split('/').last().unwrap_or("unknown"));
    println!();
    println!("Target GPU speedup: 10-50x");
    println!("Expected GPU tokens/sec: {:.2} - {:.2}",
        tokens_per_sec * 10.0, tokens_per_sec * 50.0);
    println!();

    Ok(())
}
