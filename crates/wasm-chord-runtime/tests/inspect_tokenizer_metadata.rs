/// Inspect tokenizer metadata in GGUF files
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use wasm_chord_core::GGUFParser;

fn get_model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut path = PathBuf::from(manifest_dir);
    path.pop();
    path.pop();

    let q8_path = path.join("models/tinyllama-q8.gguf");
    if q8_path.exists() {
        return q8_path;
    }

    let q4km_path = path.join("models/tinyllama-q4km.gguf");
    if q4km_path.exists() {
        q4km_path
    } else {
        path.join("models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
    }
}

#[test]
#[ignore]
fn inspect_tokenizer_metadata() {
    let model_path = get_model_path();
    println!("📂 Inspecting tokenizer metadata from: {}", model_path.display());

    let file = File::open(&model_path).expect("Failed to open model file");
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header().expect("Failed to parse header");

    println!("\n=== Tokenizer Metadata Keys ===");
    let mut keys: Vec<_> = meta.metadata.keys().collect();
    keys.sort();

    for key in keys {
        if key.contains("token") {
            let value = &meta.metadata[key];
            match value {
                wasm_chord_core::formats::gguf::MetadataValue::Array(arr) => {
                    println!("{}: Array with {} elements", key, arr.len());
                    if arr.len() <= 10 {
                        for (i, v) in arr.iter().enumerate() {
                            println!("  [{}]: {:?}", i, v);
                        }
                    } else {
                        println!("  First 5:");
                        for (i, v) in arr.iter().take(5).enumerate() {
                            println!("  [{}]: {:?}", i, v);
                        }
                        println!("  ...");
                        println!("  Last 5:");
                        for (i, v) in arr.iter().enumerate().skip(arr.len() - 5) {
                            println!("  [{}]: {:?}", i, v);
                        }
                    }
                }
                _ => {
                    println!("{}: {:?}", key, value);
                }
            }
        }
    }

    // Look specifically for BPE merges
    println!("\n=== BPE Merges ===");
    if let Some(merges) = meta.metadata.get("tokenizer.ggml.merges") {
        println!("Found tokenizer.ggml.merges!");
        if let wasm_chord_core::formats::gguf::MetadataValue::Array(arr) = merges {
            println!("  {} merge rules", arr.len());
            println!("  First 10 merges:");
            for (i, merge) in arr.iter().take(10).enumerate() {
                println!("    [{}]: {:?}", i, merge);
            }
        }
    } else {
        println!("No tokenizer.ggml.merges found");
    }

    // Check for token types
    println!("\n=== Token Types ===");
    if let Some(types) = meta.metadata.get("tokenizer.ggml.token_type") {
        println!("Found tokenizer.ggml.token_type!");
        if let wasm_chord_core::formats::gguf::MetadataValue::Array(arr) = types {
            println!("  {} token types", arr.len());
            println!("  First 20:");
            for (i, t) in arr.iter().take(20).enumerate() {
                println!("    [{}]: {:?}", i, t);
            }
        }
    } else {
        println!("No tokenizer.ggml.token_type found");
    }

    // Show some actual tokens
    println!("\n=== Sample Tokens ===");
    if let Some(wasm_chord_core::formats::gguf::MetadataValue::Array(arr)) =
        meta.metadata.get("tokenizer.ggml.tokens")
    {
        println!("First 50 tokens:");
        for (i, token) in arr.iter().take(50).enumerate() {
            if let Some(s) = token.as_string() {
                // Show bytes for special chars
                let bytes: Vec<u8> = s.bytes().collect();
                println!("  [{}]: {:?} (bytes: {:?})", i, s, bytes);
            }
        }
    }

    // Check tokenizer model type
    println!("\n=== Tokenizer Model ===");
    if let Some(model) = meta.metadata.get("tokenizer.ggml.model") {
        println!("tokenizer.ggml.model: {:?}", model);
    }
}
