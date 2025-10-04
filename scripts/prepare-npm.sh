#!/bin/bash
set -e

# Script to prepare NPM package after wasm-pack build

PKG_DIR="crates/wasm-chord-runtime/pkg"
TEMPLATE_DIR="crates/wasm-chord-runtime/pkg-template"

echo "📦 Preparing NPM package..."

# Check if pkg directory exists
if [ ! -d "$PKG_DIR" ]; then
    echo "❌ Error: $PKG_DIR not found. Run wasm-pack build first."
    exit 1
fi

# Merge package.json metadata
if [ -f "$TEMPLATE_DIR/package.json" ]; then
    echo "🔧 Merging package.json metadata..."
    node -e "
        const fs = require('fs');
        const pkg = JSON.parse(fs.readFileSync('$PKG_DIR/package.json', 'utf8'));
        const template = JSON.parse(fs.readFileSync('$TEMPLATE_DIR/package.json', 'utf8'));

        // Merge template fields into generated package.json
        const merged = {
            ...pkg,
            ...template,
            // Keep version and files from wasm-pack
            version: pkg.version,
            files: pkg.files,
            main: pkg.main,
            types: pkg.types,
            sideEffects: pkg.sideEffects
        };

        fs.writeFileSync('$PKG_DIR/package.json', JSON.stringify(merged, null, 2));
    "
fi

# Copy README
if [ -f "$TEMPLATE_DIR/README.md" ]; then
    echo "📄 Copying README.md..."
    cp "$TEMPLATE_DIR/README.md" "$PKG_DIR/"
fi

# Copy LICENSE if exists
if [ -f "LICENSE-MIT" ]; then
    echo "📜 Copying LICENSE..."
    cp LICENSE-MIT "$PKG_DIR/LICENSE"
elif [ -f "LICENSE" ]; then
    cp LICENSE "$PKG_DIR/"
fi

echo "✅ NPM package ready at $PKG_DIR"
echo ""
echo "To publish:"
echo "  cd $PKG_DIR"
echo "  npm publish --access public"
