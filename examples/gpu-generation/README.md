# GPU-Accelerated Generation Example

This example demonstrates how to enable GPU acceleration for 5-10x speedup.

## Building with GPU Support

```bash
# Build with GPU feature enabled
cargo build --release --manifest-path examples/gpu-generation/Cargo.toml

# Or build from workspace root
cargo build --release --features gpu --bin gpu-generation
```

## Usage

```bash
cargo run --release --manifest-path examples/gpu-generation/Cargo.toml path/to/model.gguf
```

## Key Differences from CPU-Only

The main difference is adding GPU initialization after creating the model:

```rust
use wasm_chord_runtime::Model;

// Create model as usual
let mut model = Model::new(config);

// Initialize GPU backend (feature-gated)
#[cfg(feature = "gpu")]
model.init_gpu()?;  // Automatically falls back to CPU if GPU unavailable

// Rest of the code is identical
model.generate(prompt, &tokenizer, &config)?;
```

## Performance

With GPU enabled:
- **Matmul operations**: 5-10x faster
- **Overall speedup**: 3-5x (depends on model size)
- **Fallback**: Automatic CPU fallback if GPU fails

## Dependencies

GPU support requires:
- `wgpu` - WebGPU implementation
- `pollster` - Async runtime for GPU initialization

These are automatically included when using the `gpu` feature flag.
