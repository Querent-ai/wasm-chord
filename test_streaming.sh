#!/bin/bash
# Test streaming chat application

echo "Testing streaming chat..."

# Send a question and quit
echo -e "What is 2+2?\nquit" | timeout 120 cargo run --release --manifest-path examples/chat-streaming/Cargo.toml
