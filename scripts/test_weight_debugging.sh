#!/usr/bin/env bash
# Test the comprehensive weight debugging tool

echo "🧪 Testing Comprehensive Weight Debugging Tool"
echo "=============================================="

# Build the project first
echo "Building project..."
cargo build --release --manifest-path examples/simple-generation/Cargo.toml

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"

# Run a simple generation test to see the debugging output
echo ""
echo "Running simple generation test with debugging..."
echo "==============================================="

# Set debug environment variables
export DEBUG_WEIGHTS=1
export DUMP_LAYER0=1

# Run the test
cargo run --release --manifest-path examples/simple-generation/Cargo.toml -- "Hello" 2>&1 | head -50

echo ""
echo "🎯 Look for these key debugging outputs:"
echo "======================================"
echo "• '🧪 Running comprehensive debugging...'"
echo "• '🔍 CHECKING ATTENTION WEIGHT ORIENTATION'"
echo "• 'WQ stats: sum=..., mean=..., variance=...'"
echo "• '⚠️  WQ has very low variance - may need transposing!' (if bug found)"
echo "• '🔧 Attempting to transpose attention weights...' (if fix applied)"
echo ""
echo "If you see the transpose message, the bug was found and fixed!"
echo "If not, the weights are already correctly oriented."
