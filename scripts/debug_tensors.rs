use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::formats::gguf::GGUFParser;
use wasm_chord_core::formats::gguf::TensorLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/tinyllama-1.1b.Q4_0.gguf";
    println!("🔍 Debugging tensor names in: {}", model_path);
    
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;
    
    println!("📊 Metadata:");
    println!("   Tensors: {}", meta.tensor_count);
    println!("   KV pairs: {}", meta.kv_count);
    
    // Create tensor loader to see what tensors are available
    let mut tensor_loader = TensorLoader::new();
    
    // Try to load output.weight
    match tensor_loader.load_tensor("output.weight", &mut parser) {
        Ok(data) => {
            println!("✅ Found 'output.weight' tensor with {} elements", data.len());
        }
        Err(e) => {
            println!("❌ 'output.weight' not found: {}", e);
        }
    }
    
    // Try other possible names
    let possible_names = ["lm_head.weight", "output.weight", "lm_head", "output"];
    for name in &possible_names {
        match tensor_loader.load_tensor(name, &mut parser) {
            Ok(data) => {
                println!("✅ Found '{}' tensor with {} elements", name, data.len());
            }
            Err(_) => {
                println!("❌ '{}' not found", name);
            }
        }
    }
    
    Ok(())
}

