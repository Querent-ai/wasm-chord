#!/bin/bash
# Test chat application with automated input

echo "Testing chat application..."

# Send a simple question and quit
echo -e "Hello\nquit" | timeout 60 cargo run --release --manifest-path examples/chat/Cargo.toml
