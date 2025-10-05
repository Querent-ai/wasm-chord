#!/bin/bash
# Test generation across different model quantizations

echo "Testing Q4_0 model..."
cargo run --release --example cli -- generate --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "Hello" --max-tokens 20 --temperature 0.0

echo -e "\n\nTesting Q4_K_M model..."
cargo run --release --example cli -- generate --model models/tinyllama-q4km.gguf --prompt "Hello" --max-tokens 20 --temperature 0.0

echo -e "\n\nTesting Q8 model..."
cargo run --release --example cli -- generate --model models/tinyllama-q8.gguf --prompt "Hello" --max-tokens 20 --temperature 0.0
