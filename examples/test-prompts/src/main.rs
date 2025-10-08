/// Test multiple prompts to evaluate model performance
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing WASM-Chord with Various Prompts");
    println!("==========================================\n");

    let model_path = "models/tinyllama-1.1b.Q4_K_M.gguf";

    // Load model once
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
    println!("✅ Model loaded\n");

    // Test prompts
    let test_cases = vec![
        ("The quick brown fox", 10),
        ("Once upon a time", 10),
        ("The capital of France is", 8),
        ("Hello world", 10),
        ("To be or not to be", 10),
        ("In a galaxy far", 10),
        ("The weather today is", 10),
        ("Machine learning is", 10),
    ];

    let config = GenerationConfig {
        max_tokens: 10,
        temperature: 0.0, // Deterministic
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    for (i, (prompt, _)) in test_cases.iter().enumerate() {
        println!("---\n📝 Test #{} Prompt: {:?}", i + 1, prompt);

        match model.generate(prompt, &tokenizer, &config) {
            Ok(result) => {
                println!("🎯 Output: {:?}\n", result);
            }
            Err(e) => {
                println!("❌ Error: {:?}\n", e);
            }
        }
    }

    Ok(())
}
