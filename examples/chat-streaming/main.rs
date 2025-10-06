/// Streaming CLI Chat Application
///
/// Real-time token-by-token streaming for responsive chat experience.
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{ChatMessage, ChatTemplate, GenerationConfig, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 WASM-Chord Streaming Chat");
    println!("============================\n");

    // Model path
    let model_path = "models/tinyllama-q8.gguf";
    println!("📂 Loading model: {}", model_path);

    // === Load Model ===
    let file = File::open(model_path)?;
    let reader = BufReader::new(file);
    let mut parser = GGUFParser::new(reader);
    let meta = parser.parse_header()?;

    let config_data = parser.extract_config().ok_or("Failed to extract config from GGUF")?;
    let config: TransformerConfig = config_data.into();
    println!("✅ Model loaded: {} layers, {} vocab", config.num_layers, config.vocab_size);

    // === Load Tokenizer ===
    let tokenizer = Tokenizer::from_gguf(&meta)?;

    // === Load Weights ===
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
    println!("✅ Ready to chat!\n");

    // Chat configuration
    let template = ChatTemplate::ChatML;
    let gen_config = GenerationConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    // Chat loop
    let mut conversation: Vec<ChatMessage> = vec![
        ChatMessage::system("You are a helpful, friendly AI assistant. Keep responses concise and relevant."),
    ];

    println!("Type 'quit' to exit, 'clear' to reset conversation.\n");

    let stdin = io::stdin();
    loop {
        // Get user input
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("\nGoodbye! 👋");
            break;
        }

        if input == "clear" {
            conversation = vec![
                ChatMessage::system("You are a helpful, friendly AI assistant. Keep responses concise and relevant."),
            ];
            println!("\n✨ Conversation cleared.\n");
            continue;
        }

        // Add user message
        conversation.push(ChatMessage::user(input));

        // Format prompt
        let prompt = template.format(&conversation)?;

        // Generate with streaming
        print!("Assistant: ");
        io::stdout().flush()?;

        let start = std::time::Instant::now();

        let mut assistant_response = String::new();

        let _result = model.generate_stream(&prompt, &tokenizer, &gen_config, |_token_id, token_text| {
            // Print token in real-time
            print!("{}", token_text);
            io::stdout().flush().ok();

            assistant_response.push_str(token_text);

            // Continue generation
            true
        })?;

        let duration = start.elapsed();

        println!("\n⏱️  ({:.1}s, {:.2} tok/s)\n",
            duration.as_secs_f32(),
            gen_config.max_tokens as f32 / duration.as_secs_f32()
        );

        // Add to conversation (extract just assistant part)
        let assistant_text = if let Some(idx) = assistant_response.rfind("<|assistant|>") {
            assistant_response[idx + 13..].trim().to_string()
        } else {
            assistant_response.trim().to_string()
        };

        conversation.push(ChatMessage::assistant(&assistant_text));
    }

    Ok(())
}
