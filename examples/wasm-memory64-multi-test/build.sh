#!/bin/bash
# Build script for WebAssembly Memory64 & Multi-Memory Test

echo "🧪 Building WebAssembly Memory64 & Multi-Memory Test"
echo "=================================================="

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build without Memory64 first
echo "📝 Building without Memory64..."
cd examples/wasm-memory64-multi-test
wasm-pack build --target web --out-dir pkg --dev

if [ $? -eq 0 ]; then
    echo "✅ Build successful (without Memory64)"
    echo "   Max memory: 4GB"
    echo "   Test in browser: python3 -m http.server 8000"
    echo "   Then open: http://localhost:8000"
else
    echo "❌ Build failed (without Memory64)"
    exit 1
fi

# Build with Memory64
echo ""
echo "📝 Building with Memory64..."
wasm-pack build --target web --out-dir pkg-memory64 --dev --features memory64

if [ $? -eq 0 ]; then
    echo "✅ Build successful (with Memory64)"
    echo "   Max memory: 16GB"
    echo "   Test in browser: python3 -m http.server 8001"
    echo "   Then open: http://localhost:8001"
else
    echo "❌ Build failed (with Memory64)"
    exit 1
fi

echo ""
echo "📊 Build Summary:"
echo "   Without Memory64: pkg/ (4GB limit)"
echo "   With Memory64: pkg-memory64/ (16GB limit)"
echo ""
echo "🌐 To test in browser:"
echo "   1. Start server: python3 -m http.server 8000"
echo "   2. Open: http://localhost:8000"
echo "   3. Try allocating >4GB to test Memory64"
echo ""
echo "💡 Browser Requirements for Memory64:"
echo "   • Chrome 119+"
echo "   • Firefox 120+"
echo "   • Safari 17+"
echo "   • Edge 119+"
