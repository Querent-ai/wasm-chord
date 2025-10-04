# @querent-ai/wasm-chord

High-performance WebAssembly LLM inference runtime with support for quantized models (GGUF format).

## Features

- 🚀 **Fast**: Optimized matmul and attention kernels with loop unrolling
- 📦 **Quantization**: Support for Q4_0, Q8_0 quantized models
- 🧠 **Modern Architectures**: LLaMA 2/3, Mistral, TinyLlama with GQA support
- 🌐 **Web-native**: Pure WebAssembly, runs in any modern browser
- 🎯 **Production-ready**: Comprehensive test coverage and CI/CD

## Installation

```bash
npm install @querent-ai/wasm-chord
```

## Quick Start

```typescript
import init, { Model, Tokenizer } from '@querent-ai/wasm-chord';

// Initialize the WASM module
await init();

// Load your GGUF model
const modelBytes = await fetch('model.gguf').then(r => r.arrayBuffer());
const model = Model.from_gguf(new Uint8Array(modelBytes));

// Create tokenizer
const tokenizer = Tokenizer.from_vocab(vocab);

// Run inference
const prompt = "Once upon a time";
const tokens = tokenizer.encode(prompt);
const output = model.generate(tokens, {
  max_tokens: 100,
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40
});

const text = tokenizer.decode(output);
console.log(text);
```

## Supported Models

- **LLaMA 2/3**: All sizes (7B, 13B, 70B)
- **Mistral 7B**: Including instruct variants
- **TinyLlama 1.1B**: Optimized for edge devices
- **Phi-2/3**: Microsoft's small language models

## Model Quantization

Supports GGUF quantized formats:
- `Q4_0`: 4-bit quantization (~4GB for 7B model)
- `Q8_0`: 8-bit quantization (~7GB for 7B model)
- `F32`: Full precision (for testing)

## Performance

**TinyLlama 1.1B (Q4_0)**:
- Single token: ~50ms (Chrome on M1)
- First token: ~100ms (includes model loading)
- Throughput: ~20 tokens/sec

**Optimizations**:
- Loop unrolling (4x elements)
- Grouped Query Attention (GQA)
- KV cache for efficient generation
- SIMD-friendly memory layout

## API Reference

### Model

```typescript
class Model {
  static from_gguf(bytes: Uint8Array): Model;
  generate(tokens: Uint32Array, options: GenOptions): Uint32Array;
  forward(tokens: Uint32Array, position: number): Float32Array;
  sample(logits: Float32Array, temperature: number, top_p: number, top_k: number): number;
}
```

### Tokenizer

```typescript
class Tokenizer {
  static from_vocab(vocab: Map<string, number>): Tokenizer;
  encode(text: string): Uint32Array;
  decode(tokens: Uint32Array): string;
}
```

### GenOptions

```typescript
interface GenOptions {
  max_tokens?: number;      // Default: 100
  temperature?: number;     // Default: 1.0 (0.0 = greedy)
  top_p?: number;          // Default: 1.0 (disabled)
  top_k?: number;          // Default: 0 (disabled)
  stop_tokens?: number[];  // Default: []
}
```

## Architecture

```
┌─────────────────────────────────────┐
│         GGUF Model File             │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      wasm-chord-core (Parser)       │
│  • GGUF metadata extraction         │
│  • Tensor lazy loading              │
│  • Q4_0/Q8_0 dequantization        │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    wasm-chord-runtime (Inference)   │
│  • Transformer layers               │
│  • Multi-head attention (GQA)       │
│  • SwiGLU FFN                       │
│  • RoPE position embeddings         │
│  • Advanced sampling (top-p/top-k)  │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      wasm-chord-cpu (Kernels)       │
│  • Optimized matmul (loop unrolled) │
│  • Softmax, ReLU, GELU             │
│  • Cache-friendly memory access     │
└─────────────────────────────────────┘
```

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 89+
- ✅ Safari 15.4+
- ✅ Edge 90+

Requires WebAssembly with:
- Multi-value returns
- Bulk memory operations
- Reference types

## Development

```bash
# Clone repo
git clone https://github.com/querent-ai/wasm-chord
cd wasm-chord

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench -p wasm-chord-cpu
cargo bench -p wasm-chord-runtime

# Build WASM package
cd crates/wasm-chord-runtime
wasm-pack build --target web --scope querent-ai
```

## License

Dual-licensed under MIT or Apache 2.0.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/querent-ai/wasm-chord/blob/main/CONTRIBUTING.md).

## Acknowledgments

Built by [Querent AI](https://querent.xyz) - Agentic AI for enterprise.

Special thanks to:
- LLaMA team for the transformer architecture
- GGML/GGUF for quantization formats
- wasm-pack for seamless Rust→WASM builds
