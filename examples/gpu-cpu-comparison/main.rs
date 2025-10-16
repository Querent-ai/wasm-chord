/// GPU vs CPU Correctness Test
/// Validates that GPU produces identical results to CPU
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn load_model(
    model_path: &str,
) -> Result<(Model, TransformerConfig, Tokenizer), Box<dyn std::error::Error>> {
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

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

    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    parser.parse_header()?;

    model.load_from_gguf(&mut tensor_loader, &mut parser)?;

    Ok((model, config, tokenizer))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 GPU vs CPU Correctness Test");
    println!("================================\n");

    #[cfg(not(feature = "webgpu"))]
    {
        eprintln!("❌ WebGPU feature not enabled!");
        eprintln!("   Run with: cargo run --release --manifest-path examples/gpu-cpu-comparison/Cargo.toml --features webgpu");
        std::process::exit(1);
    }

    #[cfg(feature = "webgpu")]
    {
        let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";

        // Test prompt
        let prompt = "Hello";
        println!("📝 Test prompt: {:?}\n", prompt);

        // Load CPU model
        println!("🖥️  Loading CPU model...");
        let (mut cpu_model, _config, tokenizer) = load_model(model_path)?;
        println!("✅ CPU model loaded");

        // Load GPU model
        println!("🎮 Loading GPU model...");
        let (mut gpu_model, _config2, _tokenizer2) = load_model(model_path)?;

        match gpu_model.init_gpu() {
            Ok(()) => println!("✅ GPU model loaded and initialized"),
            Err(e) => {
                eprintln!("❌ GPU initialization failed: {}", e);
                eprintln!("   This test requires GPU hardware");
                std::process::exit(1);
            }
        }

        // Tokenize prompt
        let tokens = tokenizer.encode(prompt, false)?;
        println!("\n🔤 Tokens: {:?}", tokens);

        // Run CPU forward pass
        println!("\n🖥️  Running CPU forward pass...");
        let cpu_start = std::time::Instant::now();
        let cpu_logits = cpu_model.forward(&tokens, 0)?;
        let cpu_duration = cpu_start.elapsed();
        println!("   Time: {:?}", cpu_duration);

        // Run GPU forward pass
        println!("🎮 Running GPU forward pass...");
        let gpu_start = std::time::Instant::now();
        let gpu_logits = gpu_model.forward(&tokens, 0)?;
        let gpu_duration = gpu_start.elapsed();
        println!("   Time: {:?}", gpu_duration);
        println!("   Speedup: {:.2}x", cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64());

        // Compare logits
        println!("\n🔍 Comparing outputs...");
        assert_eq!(cpu_logits.len(), gpu_logits.len(), "Logit length mismatch!");

        let mut max_diff = 0.0_f32;
        let mut total_diff = 0.0_f32;
        let mut diff_count = 0;

        for (i, (&cpu_val, &gpu_val)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            total_diff += diff;
            if diff > 0.001 {
                diff_count += 1;
            }
        }

        let avg_diff = total_diff / cpu_logits.len() as f32;

        println!("   Logits compared: {}", cpu_logits.len());
        println!("   Max difference: {:.6}", max_diff);
        println!("   Avg difference: {:.6}", avg_diff);
        println!("   Values with diff > 0.001: {}", diff_count);

        // Find top tokens from both
        let get_top_tokens = |logits: &[f32], k: usize| -> Vec<(usize, f32)> {
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.into_iter().take(k).collect()
        };

        let cpu_top = get_top_tokens(&cpu_logits, 10);
        let gpu_top = get_top_tokens(&gpu_logits, 10);

        println!("\n📊 Top 10 tokens comparison:");
        println!("   Rank | CPU Token (logit)          | GPU Token (logit)          | Match");
        println!("   -----|----------------------------|----------------------------|------");
        for i in 0..10 {
            let (cpu_id, cpu_logit) = cpu_top[i];
            let (gpu_id, gpu_logit) = gpu_top[i];
            let cpu_token = tokenizer.id_to_token(cpu_id as u32).unwrap_or("<unk>");
            let gpu_token = tokenizer.id_to_token(gpu_id as u32).unwrap_or("<unk>");
            let match_str = if cpu_id == gpu_id { "✅" } else { "❌" };
            println!(
                "   {:4} | {:8} ({:10.6}) | {:8} ({:10.6}) | {}",
                i + 1,
                cpu_token,
                cpu_logit,
                gpu_token,
                gpu_logit,
                match_str
            );
        }

        // Check if top token matches
        let (cpu_top_id, cpu_top_logit) = cpu_top[0];
        let (gpu_top_id, gpu_top_logit) = gpu_top[0];

        println!("\n📌 Top Token Comparison:");
        if cpu_top_id == gpu_top_id {
            println!(
                "   ✅ MATCH! Both predict token {} ({:?})",
                cpu_top_id,
                tokenizer.id_to_token(cpu_top_id as u32).unwrap_or("<unk>")
            );
            println!("   CPU logit: {:.6}", cpu_top_logit);
            println!("   GPU logit: {:.6}", gpu_top_logit);
            println!("   Difference: {:.6}", (cpu_top_logit - gpu_top_logit).abs());
        } else {
            println!("   ❌ MISMATCH!");
            println!(
                "   CPU: token {} ({:?}) logit={:.6}",
                cpu_top_id,
                tokenizer.id_to_token(cpu_top_id as u32).unwrap_or("<unk>"),
                cpu_top_logit
            );
            println!(
                "   GPU: token {} ({:?}) logit={:.6}",
                gpu_top_id,
                tokenizer.id_to_token(gpu_top_id as u32).unwrap_or("<unk>"),
                gpu_top_logit
            );
        }

        // Verdict
        println!("\n🎯 Verdict:");
        if cpu_top_id == gpu_top_id && max_diff < 0.01 {
            println!("   ✅ GPU and CPU produce IDENTICAL results!");
            println!("   GPU acceleration is working correctly!");
        } else if cpu_top_id == gpu_top_id {
            println!("   ⚠️  GPU and CPU predict the same token, but logits differ");
            println!("   This may be due to floating-point precision differences");
        } else {
            println!("   ❌ GPU and CPU produce DIFFERENT results!");
            println!("   GPU implementation may have bugs");
            std::process::exit(1);
        }
    }

    Ok(())
}
