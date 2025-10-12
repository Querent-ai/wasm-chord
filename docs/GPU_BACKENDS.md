# GPU Backend Guide

wasm-chord supports multiple GPU acceleration backends for different use cases.

## Available Backends

### 1. **Candle GPU** (Opt-in via `cuda` or `metal` features)
- **Purpose**: Native GPU acceleration for maximum performance
- **Supported**: CUDA (NVIDIA), Metal (Apple Silicon)
- **Location**: `wasm-chord-gpu` crate (CandleGpuBackend)
- **Feature Flags**: `cuda` (NVIDIA), `metal` (Apple)
- **Default**: CPU-only (no GPU dependencies)
- **Usage**: Opt-in by enabling `cuda` or `metal` feature

**Example:**
```rust
let mut model = Model::new(config);
// Candle GPU only available if cuda/metal feature enabled
#[cfg(any(feature = "cuda", feature = "metal"))]
model.init_candle_gpu()?; // Uses CUDA or Metal if available
```

**Build with specific GPU:**
```bash
# CUDA (NVIDIA)
cargo build --features cuda

# Metal (macOS/Apple Silicon)
cargo build --features metal

# CPU only (default - no GPU features)
cargo build
```

### 2. **WebGPU** (Feature: `webgpu`)
- **Purpose**: GPU acceleration in browsers and WebAssembly
- **Supported**: All platforms with WebGPU support (Chrome, Firefox, Safari)
- **Location**: `wasm-chord-gpu` crate (GpuBackend)
- **Feature Flag**: `webgpu` (or deprecated `gpu`)
- **Usage**: Requires async initialization

**Example:**
```rust
#[cfg(feature = "webgpu")]
{
    model.init_gpu()?; // WebGPU backend
}
```

**Build:**
```bash
cargo build --features webgpu
```

## Comparison

| Feature | Candle GPU (cuda/metal) | WebGPU | CPU (default) |
|---------|------------------------|--------|---------------|
| **Target** | Native | Browser/WASM | All platforms |
| **API** | CUDA, Metal | WebGPU | Candle CPU |
| **Availability** | Opt-in feature | Opt-in feature | Always available |
| **Platforms** | NVIDIA, Apple | WebGPU browsers | All |
| **Performance** | Best (native GPU) | Good (browser GPU) | Baseline |
| **Dependencies** | CUDA/Metal SDK | wgpu | Minimal |
| **Build Size** | Largest | Large | Smallest |
| **Initialization** | Sync | Async | Instant |
| **Fallback** | To CPU | To CPU | N/A |

## Usage Recommendations

### For Native Applications
Use **Candle GPU** for best performance (requires `cuda` or `metal` feature):

```rust
use wasm_chord_runtime::Model;

let mut model = Model::new(config);

// Initialize Candle GPU (only available with cuda/metal feature)
#[cfg(any(feature = "cuda", feature = "metal"))]
if let Err(e) = model.init_candle_gpu() {
    eprintln!("GPU init failed: {}, using CPU", e);
}

// Generate (works with or without GPU)
let response = model.generate(prompt, &tokenizer, &config)?;
```

**Build:**
```bash
# CUDA (NVIDIA)
cargo build --release --features cuda

# Metal (Apple Silicon)
cargo build --release --features metal

# CPU-only (default, smallest)
cargo build --release
```

### For Web Applications
Use **WebGPU**:

```javascript
import init, { WasmModel } from './pkg/wasm_chord_runtime.js';

await init();
const model = new WasmModel(modelBytes);

// Initialize WebGPU if feature is enabled
try {
    model.init_gpu();
} catch (e) {
    console.log("WebGPU unavailable, using CPU");
}

const response = model.generate(prompt);
```

### For Both Native and Web
Choose the right backend for each target:

```bash
# Native with CUDA GPU
cargo build --release --features cuda

# Native with Metal GPU
cargo build --release --features metal

# Native CPU-only (smallest, fastest compile)
cargo build --release

# WASM with WebGPU (browser GPU)
wasm-pack build --target web --features webgpu

# WASM CPU-only (smallest bundle)
wasm-pack build --target web
```

## Feature Flags Reference

### Runtime (`wasm-chord-runtime`)
- **`default`** - CPU-only (no GPU dependencies, smallest build)
- **`cuda`** - Enable Candle GPU with CUDA support (NVIDIA GPUs)
- **`metal`** - Enable Candle GPU with Metal support (Apple Silicon)
- **`webgpu`** - Enable WebGPU backend for browsers

### CPU Backend (`wasm-chord-cpu`)
- `gpu` - Enable Candle GPU re-export
- `cuda` - Enable CUDA support (NVIDIA)
- `metal` - Enable Metal support (Apple)

### GPU Backend (`wasm-chord-gpu`)
- `cuda` - Enable CUDA backend
- `metal` - Enable Metal backend

## Building for Different Targets

### Native CPU-only (Default, Smallest)
```bash
# Default build is CPU-only
cargo build --release
```

### Native with CUDA GPU
```bash
# Enable CUDA for NVIDIA GPUs
cargo build --release --features cuda
```

### Native with Metal GPU
```bash
# Enable Metal for Apple Silicon
cargo build --release --features metal
```

### WASM CPU-only (Default, Smallest Bundle)
```bash
# Default WASM build is CPU-only
wasm-pack build --target web
```

### WASM with WebGPU
```bash
# Enable WebGPU for browser GPU acceleration
wasm-pack build --target web --features webgpu
```

## Examples

### Native CPU-only
```bash
cargo run --release --manifest-path examples/test-capital-cpu/Cargo.toml
```

### Native with Candle GPU
```bash
# CUDA
cargo run --release --features cuda --manifest-path examples/test-capital-gpu/Cargo.toml

# Metal
cargo run --release --features metal --manifest-path examples/test-capital-gpu/Cargo.toml
```

### WebAssembly with WebGPU
```bash
cd crates/wasm-chord-runtime
wasm-pack build --target web --features webgpu
```

## Implementation Details

### Candle GPU (CandleGpuBackend)
- Located in: `crates/wasm-chord-gpu/src/candle_backend.rs`
- Device selection: CUDA → Metal → CPU (automatic)
- Operations: matmul, RMS norm, attention, SiLU, RoPE
- Tensor conversion: f32 ↔ Candle Tensor

### WebGPU (GpuBackend)
- Located in: `crates/wasm-chord-gpu/src/lib.rs`
- Uses wgpu for WebGPU API
- Shader-based computation
- WGSL shaders for kernels

## Migration from `gpu` to `webgpu`

The `gpu` feature has been renamed to `webgpu` for clarity:

**Old:**
```toml
wasm-chord-runtime = { features = ["gpu"] }
```

**New:**
```toml
wasm-chord-runtime = { features = ["webgpu"] }
```

The old `gpu` feature still works as an alias but will be removed in a future version.

## Troubleshooting

### "GPU initialization failed"
- Candle GPU: Check CUDA/Metal drivers installed
- WebGPU: Check browser WebGPU support

### "No GPU available"
- Normal behavior, falls back to CPU automatically
- Not an error, just informational

### Performance Issues
1. Check GPU is actually being used (not CPU fallback)
2. Verify correct feature flags enabled
3. Use release build (`--release`)
4. Check GPU drivers are up to date

## Performance Benchmarks

Approximate inference speed (tokens/sec) for TinyLLaMA 1.1B:

| Backend | Speed | Notes |
|---------|-------|-------|
| CPU (Native) | 5-15 | Candle-optimized |
| Candle GPU (CUDA) | 50-100 | NVIDIA GPU |
| Candle GPU (Metal) | 40-80 | Apple Silicon |
| WebGPU (Browser) | 10-30 | Varies by browser |
| CPU (WASM) | 2-8 | Browser fallback |

*Benchmarks are approximate and vary by hardware*
