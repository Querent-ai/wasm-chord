# wasm-chord

> High-performance LLM inference runtime for WebAssembly and native platforms

**wasm-chord** is a production-grade inference runtime for Large Language Models (LLMs), designed for deployment across WebAssembly, native, and server environments. Built with Rust, it provides efficient execution of quantized models with GPU acceleration support through CUDA, Metal, and WebGPU.

[![CI](https://github.com/querent-ai/wasm-chord/workflows/CI/badge.svg)](https://github.com/querent-ai/wasm-chord/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

## Overview

wasm-chord enables efficient LLM inference across diverse deployment targets:

- **Browser**: Run models client-side with WebGPU acceleration
- **Native**: CUDA/Metal GPU support for desktop and server deployments
- **WebAssembly**: WASI-compatible runtime for edge computing
- **Mobile**: Compile to native targets for iOS/Android

The runtime supports the GGUF model format with multiple quantization schemes (Q4_K, Q5_K, Q8_K) and implements optimized transformer operations including RoPE, multi-head attention with KV caching, and SwiGLU feed-forward networks.

## Features

### Core Runtime
- **Multiple Backend Support**: CUDA (NVIDIA), Metal (Apple Silicon), WebGPU (browsers), CPU (SIMD-optimized)
- **GGUF Format**: Native support for GGUF model files from llama.cpp ecosystem
- **Quantization**: Q4_K, Q5_K, Q8_K quantization schemes
- **Streaming Inference**: Token-by-token generation with callback support
- **KV Caching**: Efficient attention caching for improved throughput
- **Chat Templates**: Built-in support for ChatML, Llama2, and Alpaca templates

### Performance
- **GPU Acceleration**: WebGPU shaders for browser, Candle backend for CUDA/Metal
- **Optimized Kernels**: SIMD CPU operations with Rayon parallelism
- **Memory Efficient**: Quantized weights with on-the-fly dequantization
- **Batched Operations**: Optimized matrix operations via Candle and gemm crates

### Production Ready
- **Stable ABI**: C-compatible interface for host language integration
- **Type Safety**: Rust's ownership system prevents memory errors
- **Comprehensive Testing**: Unit tests, integration tests, and CI pipeline
- **Cross-platform**: Linux, macOS, Windows, and WASI environments

## Quick Start

### Native (Rust)

Add to your `Cargo.toml`:

```toml
[dependencies]
wasm-chord-core = "0.1"
wasm-chord-runtime = "0.1"
```

Basic usage:

```rust
use wasm_chord_core::{GGUFParser, TensorLoader, Tokenizer};
use wasm_chord_runtime::{Model, TransformerConfig, GenerationConfig};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let file = File::open("model.gguf")?;
    let mut parser = GGUFParser::new(file);
    let meta = parser.parse_header()?;

    // Extract config
    let config_data = parser.extract_config().unwrap();
    let config: TransformerConfig = config_data.into();

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&meta)?;

    // Initialize model
    let mut model = Model::new(config);
    let mut tensor_loader = TensorLoader::new(parser.tensor_data_offset()?);

    // Register and load weights
    for tensor in meta.tensors.iter() {
        tensor_loader.register_tensor(
            tensor.name.clone(),
            tensor.clone(),
            tensor.offset,
        );
    }
    model.load_from_gguf(&mut tensor_loader, &mut parser)?;

    // Generate text
    let gen_config = GenerationConfig {
        max_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    let response = model.generate("Hello, how are you?", &tokenizer, &gen_config)?;
    println!("{}", response);

    Ok(())
}
```

### With GPU Acceleration

Enable CUDA or Metal:

```bash
# CUDA (NVIDIA)
cargo build --release --features cuda

# Metal (Apple Silicon)
cargo build --release --features metal
```

Initialize GPU backend:

```rust
#[cfg(any(feature = "cuda", feature = "metal"))]
model.init_candle_gpu()?;
```

### Browser (WebAssembly)

Build for web:

```bash
cd crates/wasm-chord-runtime
wasm-pack build --target web --features webgpu
```

JavaScript usage:

```javascript
import init, { WasmModel } from './pkg/wasm_chord_runtime.js';

await init();

// Load model (as Uint8Array)
const modelBytes = await fetch('model.gguf').then(r => r.arrayBuffer());
const model = new WasmModel(new Uint8Array(modelBytes));

// Configure generation
model.set_config(50, 0.7, 0.9, 40, 1.1);

// Generate with streaming
model.generate_stream("Hello, how are you?", (token) => {
    console.log(token);
    return true; // continue generation
});
```

## Architecture

### Crate Structure

```
wasm-chord/
├── crates/
│   ├── wasm-chord-core/       # Core tensor operations, GGUF parsing, tokenization
│   ├── wasm-chord-runtime/    # Model implementation, inference engine, C ABI
│   ├── wasm-chord-gpu/        # GPU backends (Candle, WebGPU)
│   └── wasm-chord-cpu/        # CPU backend with SIMD optimizations
└── examples/                  # Usage examples and benchmarks
```

### Backend Selection

The runtime automatically selects the best available backend:

1. **Candle GPU** (CUDA/Metal) - Native GPU, highest performance
2. **WebGPU** - Browser GPU acceleration
3. **CPU** - SIMD-optimized fallback (always available)

### Feature Flags

| Feature | Description | Use Case |
|---------|-------------|----------|
| `default` | CPU-only | Smallest build, universal compatibility |
| `cuda` | CUDA GPU support | NVIDIA GPUs on Linux/Windows |
| `metal` | Metal GPU support | Apple Silicon Macs |
| `webgpu` | WebGPU support | Browser deployment with GPU |

## Supported Models

wasm-chord supports models in GGUF format with the following architectures:

- **LLaMA** (LLaMA 1, LLaMA 2, LLaMA 3)
- **Mistral** (Mistral 7B, Mixtral)
- **TinyLlama** (1.1B parameters)
- **Phi** (Microsoft Phi-2, Phi-3)

Quantization support:
- Q4_K_M (4-bit, medium quality)
- Q5_K_M (5-bit, high quality)
- Q8_0 (8-bit, highest quality)

## Examples

The repository includes numerous examples demonstrating various use cases:

### Basic Generation
```bash
# Simple text generation
cargo run --release --example simple-generation

# With CUDA GPU
cargo run --release --features cuda --example simple-generation

# Chat interface
cargo run --release --example chat

# Streaming output
cargo run --release --example chat-streaming
```

### Advanced Usage
```bash
# Performance benchmarking
cargo run --release --example benchmark

# GPU vs CPU comparison
cargo run --release --features cuda --example gpu-cpu-comparison

# Browser example
cd examples/wasm-capital-test
wasm-pack build --target web
python -m http.server 8000
```

## Roadmap

### Completed

- [x] GGUF format parsing and tensor loading
- [x] Core transformer implementation (attention, FFN, RMS norm)
- [x] Multiple quantization schemes (Q4_K, Q5_K, Q8_K)
- [x] RoPE positional embeddings
- [x] KV cache with efficient memory management
- [x] Streaming token generation
- [x] Chat template support (ChatML, Llama2, Alpaca)
- [x] CPU backend with Candle optimization
- [x] GPU backends (CUDA, Metal via Candle)
- [x] WebGPU backend for browsers
- [x] Tokenizer integration (BPE, SentencePiece)
- [x] C ABI for host integration
- [x] WebAssembly bindings (wasm-bindgen)
- [x] Comprehensive test suite
- [x] CI/CD pipeline

### In Progress

- [ ] Memory64 support for >4GB models in WASM
- [ ] Multi-memory sharding for large models
- [ ] Fused kernel optimizations (dequant+GEMM)
- [ ] Flash Attention implementation
- [ ] Speculative decoding

### Planned

- [ ] Model quantization utilities
- [ ] Fine-tuning support
- [ ] ONNX format support
- [ ] Python bindings (PyO3)
- [ ] Additional model architectures (GPT-J, Falcon, Bloom)
- [ ] Distributed inference across multiple devices
- [ ] Model hub integration
- [ ] Profiling and debugging tools

## Performance

Performance characteristics on TinyLLaMA 1.1B Q4_K_M:

| Configuration | Tokens/sec | First Token Latency | Memory Usage |
|--------------|------------|---------------------|--------------|
| CUDA GPU (RTX 3090) | 80-100 | ~150ms | 1.2 GB |
| Metal GPU (M1 Max) | 60-80 | ~200ms | 1.5 GB |
| CPU Native (16 cores) | 15-25 | ~500ms | 2.0 GB |
| WebGPU (Chrome) | 20-35 | ~800ms | 2.5 GB |
| WASM CPU | 5-10 | ~2000ms | 2.5 GB |

*Benchmarks are approximate and vary by hardware configuration*

## Development

### Prerequisites

- Rust 1.75 or later
- For CUDA: CUDA Toolkit 11.8+
- For Metal: macOS 12.0+ with Xcode Command Line Tools
- For WebGPU: wasm-pack (`cargo install wasm-pack`)

### Building

```bash
# Clone repository
git clone https://github.com/querent-ai/wasm-chord
cd wasm-chord

# Install WASM target
rustup target add wasm32-unknown-unknown

# Build all crates
cargo build --workspace --release

# Run tests
cargo test --workspace

# Run clippy
cargo clippy --workspace --lib -- -D warnings

# Build WASM
cd crates/wasm-chord-runtime
wasm-pack build --target web
```

### Testing

The project includes extensive tests:

```bash
# Unit tests
cargo test --package wasm-chord-core

# Integration tests
cargo test --package wasm-chord-runtime

# With specific features
cargo test --features cuda

# With backtrace
RUST_BACKTRACE=1 cargo test
```

### Project Structure

```
wasm-chord/
├── crates/
│   ├── wasm-chord-core/
│   │   ├── src/
│   │   │   ├── tensor.rs          # Tensor primitives
│   │   │   ├── formats/gguf.rs    # GGUF parser
│   │   │   ├── quant.rs           # Quantization schemes
│   │   │   └── tokenizer.rs       # BPE/SentencePiece
│   │   └── Cargo.toml
│   ├── wasm-chord-runtime/
│   │   ├── src/
│   │   │   ├── transformer/       # Model implementation
│   │   │   ├── inference.rs       # Inference engine
│   │   │   ├── sampling.rs        # Token sampling
│   │   │   ├── chat.rs            # Chat templates
│   │   │   ├── web.rs             # WASM bindings
│   │   │   └── abi.rs             # C ABI
│   │   └── Cargo.toml
│   ├── wasm-chord-gpu/
│   │   ├── src/
│   │   │   ├── candle_backend.rs  # Candle GPU (CUDA/Metal)
│   │   │   ├── lib.rs             # WebGPU backend
│   │   │   └── *.wgsl             # WebGPU shaders
│   │   └── Cargo.toml
│   └── wasm-chord-cpu/
│       ├── src/
│       │   ├── backend.rs         # Candle CPU backend
│       │   └── lib.rs
│       └── Cargo.toml
├── examples/                      # Usage examples
├── docs/                          # Documentation
└── .github/workflows/             # CI configuration
```

## Documentation

- [GPU Backend Guide](docs/GPU_BACKENDS.md) - Comprehensive guide to GPU acceleration options
- [API Documentation](https://docs.rs/wasm-chord-runtime) - Generated API docs (coming soon)
- [Examples](examples/) - Practical usage examples

## Contributing

Contributions are welcome! Areas where we'd appreciate help:

- Performance optimization (kernel implementations, memory management)
- Additional model architecture support
- Documentation and examples
- Testing on diverse hardware configurations
- Bug reports and feature requests

Please open an issue before starting major work to discuss the approach.

### Development Guidelines

1. Run tests before submitting: `cargo test --workspace`
2. Ensure clippy passes: `cargo clippy --workspace -- -D warnings`
3. Format code: `cargo fmt --all`
4. Update documentation for public APIs
5. Add tests for new functionality

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Acknowledgments

This project builds upon excellent work from the broader community:

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format and quantization techniques
- [Candle](https://github.com/huggingface/candle) - ML framework with GPU support
- [wgpu](https://github.com/gfx-rs/wgpu) - WebGPU implementation
- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) - Rust/WASM interop

## Citation

If you use wasm-chord in your research or project, please cite:

```bibtex
@software{wasm_chord,
  title = {wasm-chord: High-performance LLM inference runtime},
  author = {Querent AI},
  year = {2024},
  url = {https://github.com/querent-ai/wasm-chord}
}
```

---

**Developed by [Querent AI](https://querent.xyz)**
