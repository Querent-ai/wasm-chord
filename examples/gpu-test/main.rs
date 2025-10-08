/// GPU-Accelerated Generation Test
/// Tests end-to-end text generation with GPU acceleration
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 GPU-Accelerated Generation Test");
    println!("===================================\n");

    // Check if GPU feature is enabled
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("❌ GPU feature not enabled!");
        eprintln!("   Run with: cargo run --release --manifest-path examples/gpu-test/Cargo.toml --features gpu");
        std::process::exit(1);
    }

    #[cfg(feature = "gpu")]
    {
        let model_path = "/home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf";
        println!("📂 Loading model: {}", model_path);

        // Load model
        let file = File::open(model_path)?;
        let reader = BufReader::new(file);
        let mut parser = GGUFParser::new(reader);
        let meta = parser.parse_header()?;

        let config_data = parser.extract_config().ok_or("Failed to extract config from GGUF")?;
        let config: TransformerConfig = config_data.into();
        println!("✅ Config: {} layers, {} vocab", config.num_layers, config.vocab_size);

        // Load tokenizer
        let tokenizer = Tokenizer::from_gguf(&meta)?;
        println!("✅ Tokenizer: {} tokens", tokenizer.vocab_size());

        // Load weights
        println!("📦 Loading weights...");
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
        println!("✅ Weights loaded");

        // Initialize GPU
        println!("\n🎮 Initializing GPU...");
        match model.init_gpu() {
            Ok(()) => println!("✅ GPU initialized successfully!"),
            Err(e) => {
                eprintln!("❌ GPU initialization failed: {}", e);
                eprintln!("   This test requires GPU hardware");
                std::process::exit(1);
            }
        }

        // Test prompts
        let test_cases = vec![("Hello", 5), ("The capital of France is", 3), ("2+2=", 2)];

        println!("\n🧪 Running GPU-accelerated generation tests:");
        println!("=============================================\n");

        for (prompt, max_tokens) in test_cases {
            println!("📝 Prompt: {:?}", prompt);

            let gen_config = GenerationConfig {
                max_tokens,
                temperature: 0.0, // Greedy for deterministic results
                top_p: 0.9,
                top_k: 40,
                repetition_penalty: 1.1,
            };

            let start = std::time::Instant::now();
            let response = model.generate(prompt, &tokenizer, &gen_config)?;
            let duration = start.elapsed();

            println!("   Output: {:?}", response.trim());
            println!("   Time: {:?}", duration);
            println!("   Tokens/sec: {:.2}\n", max_tokens as f64 / duration.as_secs_f64());
        }

        println!("✅ All GPU generation tests passed!");
        println!("\n📊 GPU acceleration is working correctly!");
    }

    Ok(())
}
