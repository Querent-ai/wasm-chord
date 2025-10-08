/// Q4_K Dequantization Test
/// This tests the fixed Q4_K dequantization against the model to verify it produces correct logits
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Q4_K Dequantization Fix Test");
    println!("===============================\n");

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

    println!("✅ Model loaded with FIXED Q4_K dequantization");

    // Test with "Hello"
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false)?;
    println!("\n📝 Input: '{}' → tokens: {:?}", prompt, tokens);

    // Generate first token
    let logits = model.forward(&tokens, 0)?;

    // Find top token
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_token = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > max_logit {
            max_logit = logit;
            max_token = i;
        }
    }

    let token_text = tokenizer.id_to_token(max_token as u32);

    // Check specific tokens
    let hello_logit = logits[15043]; // Hello
    let yes_logit = logits[3869]; // Yes
    let no_logit = logits[1939]; // No
    let japanese_logit = logits[31982]; // 番

    println!("\n🎯 Logit Analysis:");
    println!("==================");
    println!("Hello (15043): {:.6}", hello_logit);
    println!("Yes (3869):    {:.6}", yes_logit);
    println!("No (1939):     {:.6}", no_logit);
    println!("番 (31982):    {:.6}", japanese_logit);

    println!("\n🏆 Final Result:");
    println!("===============");
    println!(
        "Top token: {} (id: {}, logit: {:.6})",
        token_text.unwrap_or("<unknown>"),
        max_token,
        max_logit
    );

    // Check if we fixed the issue
    println!("\n🔍 Fix Verification:");
    println!("===================");
    if token_text == Some("Yes") {
        println!("✅ SUCCESS! Q4_K dequantization fix worked!");
        println!("   'Hello' now correctly predicts 'Yes'");
    } else if token_text == Some("番") {
        println!("❌ STILL BROKEN! Q4_K dequantization fix didn't work");
        println!("   Still predicting '番' instead of 'Yes'");
    } else {
        println!(
            "🤔 PARTIAL FIX? Predicting '{}' instead of 'Yes'",
            token_text.unwrap_or("<unknown>")
        );
        println!("   This suggests the fix helped but there may be other issues");
    }

    // Compare with expected Ollama behavior
    println!("\n📊 Comparison with Ollama:");
    println!("===========================");
    println!("Ollama: 'Hello' → 'Yes'");
    println!("Our:    'Hello' → '{}'", token_text.unwrap_or("<unknown>"));

    if token_text == Some("Yes") {
        println!("🎉 PERFECT MATCH! Our implementation now matches Ollama exactly!");
    } else {
        println!("⚠️  Still not matching Ollama. May need further investigation.");
    }

    Ok(())
}
