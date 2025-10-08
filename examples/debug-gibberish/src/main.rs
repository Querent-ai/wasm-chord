/// Debug tool to identify why we're getting gibberish
/// Compares tokenization and first-token logits with llama.cpp
use std::fs::File;
use std::io::BufReader;
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Gibberish Debug Tool");
    println!("======================\n");

    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

    // Load model
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config")?;
    let config: TransformerConfig = config_data.into();

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;

    println!("📋 Testing Tokenization");
    println!("=======================\n");

    // Test 1: Raw prompt
    let raw_prompt = "Hello there";
    println!("1️⃣  Raw prompt: {:?}", raw_prompt);
    let tokens_raw = tokenizer.encode(raw_prompt, false)?;
    println!("   Tokens: {:?}", tokens_raw);
    println!("   Count: {}", tokens_raw.len());
    for (i, &token_id) in tokens_raw.iter().enumerate() {
        let decoded = tokenizer.decode(&[token_id], false)?;
        println!("   [{}] {} -> {:?}", i, token_id, decoded);
    }

    // Test 2: With BOS
    println!("\n2️⃣  With BOS token:");
    let tokens_bos = tokenizer.encode(raw_prompt, true)?; // add_bos=true
    println!("   Tokens: {:?}", tokens_bos);
    println!("   Count: {}", tokens_bos.len());
    for (i, &token_id) in tokens_bos.iter().enumerate() {
        let decoded = tokenizer.decode(&[token_id], false)?;
        println!("   [{}] {} -> {:?}", i, token_id, decoded);
    }

    // Test 3: Chat template
    let chat_prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\nHello there</s>\n<|assistant|>\n";
    println!("\n3️⃣  Chat template: {:?}", chat_prompt);
    let tokens_chat = tokenizer.encode(chat_prompt, false)?;
    println!("   Tokens: {:?}", tokens_chat);
    println!("   Count: {}", tokens_chat.len());
    for (i, &token_id) in tokens_chat.iter().take(20).enumerate() {
        let decoded = tokenizer.decode(&[token_id], false)?;
        println!("   [{}] {} -> {:?}", i, token_id, decoded);
    }

    // Test 4: Check special tokens
    println!("\n4️⃣  Special tokens:");
    let special = tokenizer.special_tokens();
    println!("   BOS: {}", special.bos_token_id);
    println!("   EOS: {}", special.eos_token_id);
    println!("   UNK: {}", special.unk_token_id);

    // Decode special tokens
    let bos_decoded = tokenizer.decode(&[special.bos_token_id], false)?;
    let eos_decoded = tokenizer.decode(&[special.eos_token_id], false)?;
    println!("   BOS decoded: {:?}", bos_decoded);
    println!("   EOS decoded: {:?}", eos_decoded);

    // Test 5: Load model and get first-token logits
    println!("\n5️⃣  First-token logits test:");
    println!("   Loading model weights...");

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
    println!("   ✅ Weights loaded");

    // Test with just "Hello there" (with BOS)
    println!("\n   Testing prompt: {:?} (with BOS)", raw_prompt);
    let test_tokens = tokenizer.encode(raw_prompt, true)?;
    println!("   Tokens: {:?}", test_tokens);

    // Forward pass on first token only
    let first_token = &test_tokens[0..1];
    println!("   Forward pass on first token: {}", first_token[0]);

    let logits = model.forward(first_token, 0)?;
    let last_logits = &logits[(logits.len() - config.vocab_size)..];

    // Get top 10 logits
    let mut indexed: Vec<(usize, f32)> =
        last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n   Top 10 logits after first token:");
    for (i, (token_id, logit)) in indexed.iter().take(10).enumerate() {
        let decoded = tokenizer.decode(&[*token_id as u32], false)?;
        println!("   [{:2}] token {:5} = {:8.4} -> {:?}", i, token_id, logit, decoded);
    }

    println!("\n📝 Next Steps:");
    println!("   1. Compare token IDs above with llama.cpp output");
    println!("   2. Run: echo 'Hello there' | /home/puneet/llama.cpp/llama-cli -m {} --log-disable -n 0 2>/dev/null", model_path);
    println!("   3. Check if BOS token ({}) is being used correctly", special.bos_token_id);
    println!("   4. Verify top logits match between implementations");

    Ok(())
}
