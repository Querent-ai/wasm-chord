#!/bin/bash

# Build script for @querent/wasm-chord packages
# Tests both web and node package builds

set -e

echo "🚀 Building @querent/wasm-chord packages..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf /home/puneet/wasm-chord/bindings/js/pkg /home/puneet/wasm-chord/bindings/js/pkg-node /home/puneet/wasm-chord/bindings/js/dist

# Build web package
echo "🌐 Building web package..."
cd /home/puneet/wasm-chord/bindings/js
cp package-web.json package.json
npm run build:wasm
echo "✅ Web WASM build complete"

# Build node package  
echo "🖥️ Building node package..."
cp package-node.json package.json
npm run build:wasm
echo "✅ Node WASM build complete"

# Test builds
echo "🧪 Testing builds..."

# Test web package
echo "Testing web package..."
if [ -f "pkg/wasm_chord_runtime.js" ]; then
    echo "✅ Web package: wasm_chord_runtime.js found"
else
    echo "❌ Web package: wasm_chord_runtime.js missing"
    exit 1
fi

# Test node package
echo "Testing node package..."
if [ -f "pkg-node/wasm_chord_runtime.js" ]; then
    echo "✅ Node package: wasm_chord_runtime.js found"
else
    echo "❌ Node package: wasm_chord_runtime.js missing"
    exit 1
fi

# Check file sizes
echo "📊 Package sizes:"
echo "Web WASM: $(du -h pkg/wasm_chord_runtime_bg.wasm | cut -f1)"
echo "Node WASM: $(du -h pkg-node/wasm_chord_runtime_bg.wasm | cut -f1)"

echo "🎉 All packages built successfully!"
echo ""
echo "📦 Ready for publishing:"
echo "  - Web package: bindings/js/pkg/"
echo "  - Node package: bindings/js/pkg-node/"
echo ""
echo "📝 Next steps:"
echo "  1. npm publish (web package)"
echo "  2. npm publish (node package)"
echo "  3. Update documentation"
