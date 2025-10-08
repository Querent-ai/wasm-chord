# 🎵 wasm-chord

> Privacy-first LLM inference runtime for WebAssembly

**wasm-chord** is a production-grade Rust → WebAssembly runtime for executing quantized Large Language Models (LLMs) in browser and WASI environments. Built by [Querent AI](https://querent.xyz), it provides a stable, minimal ABI for host languages, WebGPU acceleration with CPU fallback, and support for model streaming and caching.

[![CI](https://github.com/querent-ai/wasm-chord/workflows/CI/badge.svg)](https://github.com/querent-ai/wasm-chord/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

---

## ✨ Features

- **🔒 Privacy-First**: 100% client-side inference, zero server calls
- **🌐 Universal**: Runs in browsers, Node.js, Wasmtime, Wasmer, and WASI runtimes
- **⚡ Fast**: WebGPU compute backend with SIMD CPU fallback
- **📦 Quantized Models**: Supports GGUF format (Q4, Q8 quantization)
- **🔄 Token Streaming**: Real-time streaming inference with async iterators
- **💾 Persistent Caching**: IndexedDB (browser) and filesystem (Node/WASI) caching
- **🎯 Deterministic**: Optional deterministic execution for reproducibility
- **🛡️ Sandboxed**: Secure execution with WebAssembly isolation guarantees

---

## 🚀 Quick Start

### Browser (JavaScript/TypeScript)

```bash
npm install @querent/wasm-chord
```

```typescript
import { WasmChord } from '@querent/wasm-chord';

// Initialize runtime
const runtime = await WasmChord.init({ gpuEnabled: true });

// Load model
const model = await runtime.loadModel('/models/llama-7b-q4.gguf');

// Streaming inference
for await (const token of model.inferStream('Hello, how are you?')) {
  console.log(token);
}

// Blocking inference
const response = await model.infer('Explain quantum computing', {
  maxTokens: 512,
  temperature: 0.7,
});
```

### Rust (Native)

```toml
[dependencies]
wasm-chord-runtime = "0.1"
```

```rust
use wasm_chord_runtime::Runtime;

fn main() {
    let mut rt = Runtime::new();
    let model = rt.load_model("llama-7b-q4.gguf").unwrap();

    let response = rt.infer(model, "Hello from Rust!").unwrap();
    println!("{}", response);
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Host (Browser / Node / WASI)     │
│              JS/TS or Rust bindings      │
└─────────────────┬───────────────────────┘
                  │ C ABI / WIT Interface
┌─────────────────▼───────────────────────┐
│         wasm-chord Runtime (Wasm)        │
│  ┌──────────────┬────────────────────┐  │
│  │ Core Engine  │  Backend Selection │  │
│  │  - GGUF      │   ┌──────┬──────┐ │  │
│  │  - Tensors   │   │ GPU  │ CPU  │ │  │
│  │  - Quant     │   │(WGPU)│(SIMD)│ │  │
│  └──────────────┴───┴──────┴──────┴─┘  │
└─────────────────────────────────────────┘
```

### Crates

- **`wasm-chord-core`**: Tensor primitives, GGUF parser, quantization
- **`wasm-chord-runtime`**: Wasm runtime, C ABI, model lifecycle
- **`wasm-chord-gpu`**: WebGPU compute shaders and kernels
- **`wasm-chord-cpu`**: SIMD CPU kernels with rayon parallelism
- **`bindings/js`**: TypeScript/JavaScript wasm-bindgen wrapper

---

## 📋 Roadmap

### Phase 1 - MVP ✅ (Current)
- [x] Cargo workspace scaffold
- [x] GGUF streaming parser
- [x] CPU GEMM kernels
- [x] C ABI exports
- [x] WIT interface definitions
- [x] JS bindings scaffold
- [x] Web demo example

### Phase 2 - Core Features ✅
- [x] WebGPU backend implementation
- [x] Token streaming API
- [x] Tokenizer integration (BPE/SentencePiece)
- [x] Model caching (IndexedDB/FS)
- [x] Memory64 support

### Phase 3 - Optimization
- [ ] Multi-memory layout
- [ ] Attention KV cache
- [ ] Fused kernels (dequant+GEMM)
- [ ] Layer sharding for large models

### Phase 4 - Ecosystem
- [ ] Python bindings (via Pyodide)
- [ ] VSCode extension
- [ ] Example integrations (Obsidian, Notion, etc.)
- [ ] ONNX format support

---

## 🔧 Development

### Prerequisites

- Rust 1.75+ with `wasm32-unknown-unknown` target
- Node.js 18+ (for JS bindings)
- wasm-pack (for building wasm modules)

### Build

```bash
# Install wasm32 target
rustup target add wasm32-unknown-unknown

# Build all crates
cargo build --workspace

# Build for wasm
cargo build --target wasm32-unknown-unknown --package wasm-chord-runtime

# Build JS bindings
cd crates/wasm-chord-runtime
wasm-pack build --target web
```

### Test

```bash
# Run all tests
cargo test --workspace

# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Test specific crate
cargo test -p wasm-chord-core
```

### Web Demo

```bash
# Serve the example
cd examples/web-demo
python -m http.server 8000

# Open http://localhost:8000
```

---

## 🎯 Design Goals

1. **Portability**: Single codebase runs everywhere WebAssembly does
2. **Performance**: Competitive with native inference on commodity hardware
3. **Security**: No arbitrary host access; sandboxed execution
4. **Determinism**: Reproducible results for auditing and debugging
5. **Extensibility**: Plugin points for new formats, backends, and models

---

## 📊 Benchmarks

Preliminary targets (7B quantized int8 model):

| Platform | First Token | Throughput | Memory |
|----------|-------------|------------|--------|
| Chrome (M1 Mac, WebGPU) | ~300ms | 40-60 tps | 6 GB |
| Chrome (x86, CPU) | ~800ms | 10-20 tps | 8 GB |
| Node.js (native) | ~200ms | 60-80 tps | 5 GB |

*Benchmarks are preliminary and will be updated as optimization progresses.*

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas We Need Help

- [ ] WebGPU shader optimization
- [ ] SIMD kernel tuning
- [ ] Documentation and examples
- [ ] Model format support (ONNX, SafeTensors)
- [ ] Testing on diverse hardware

---

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

## 🙏 Acknowledgments

- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and [GGML](https://github.com/ggerganov/ggml)
- Built with [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) and [wgpu](https://github.com/gfx-rs/wgpu)
- Developed by [Querent AI](https://querent.xyz)

---

**Built with ❤️ by [Querent AI](https://querent.xyz) • [Docs](https://docs.querent.xyz/wasm-chord) • [Discord](https://discord.gg/querent)**
