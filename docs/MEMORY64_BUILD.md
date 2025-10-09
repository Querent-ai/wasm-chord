# WebAssembly Memory64 Build Configuration

# This file shows how to properly build with Memory64 support

# 1. Enable Memory64 in Cargo.toml
# [features]
# memory64 = ["web-sys/memory64"]

# 2. Build with Memory64 enabled
# cargo build --release --features memory64 --target wasm32-unknown-unknown

# 3. Use wasm-pack with Memory64
# wasm-pack build --target web --features memory64

# 4. For development, use wasm-bindgen directly
# wasm-bindgen --target web --out-dir pkg --features memory64

# Note: Memory64 requires:
# - Browser support (Chrome 119+, Firefox 120+, Safari 17+)
# - Runtime support (Node.js 20+, Deno 1.40+)
# - WebAssembly memory64 proposal enabled

# Current browser support status:
# - Chrome: ✅ Supported (119+)
# - Firefox: ✅ Supported (120+) 
# - Safari: ✅ Supported (17+)
# - Edge: ✅ Supported (119+)

# Performance considerations:
# - 64-bit pointers are slower than 32-bit
# - Only use Memory64 when you need >4GB memory
# - Consider quantization for large models instead
