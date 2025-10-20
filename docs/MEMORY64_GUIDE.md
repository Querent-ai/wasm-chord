# Memory64 Guide

## Overview

Memory64 enables loading and running large language models (>4GB) that exceed standard WebAssembly's 4GB memory limit. This feature is powered by Wasmtime's Memory64 support and provides on-demand layer loading with intelligent caching.

## Architecture

### Two-Part System

#### 1. **Host Runtime** (`memory64-host`)
- Runs natively outside WASM
- Uses Wasmtime to manage Memory64 regions
- Stores model weights in host memory (can exceed 4GB)
- Exposes FFI functions for WASM to call

#### 2. **WASM FFI** (`memory64-wasm`)
- Runs inside WASM
- Imports and calls host FFI functions
- Provides safe Rust wrappers
- Manages layer caching and access

### Memory Flow

```
GGUF File (4-32GB)
        ↓
Memory64 Host Storage
        ↓
   FFI Bridge
        ↓
WASM Memory (layers on-demand)
```

## When to Use Memory64

### ✅ Use Memory64 for:
- **Large models** (>4GB): Llama-2-7B, Mistral-7B, Llama-2-13B, etc.
- **Native applications**: Using Wasmtime or Wasmer
- **Server-side inference**: Node.js with native bindings
- **Desktop applications**: Electron, Tauri with native modules

### ❌ Don't use Memory64 for:
- **Browser environments**: Memory64 not supported, use standard loading
- **Small models** (<3GB): Standard loading is faster
- **WASM without host runtime**: Pure WASM environments

## Automatic Threshold Detection

wasm-chord automatically enables Memory64 based on model size:

```rust
const MEMORY64_THRESHOLD: u64 = 3_000_000_000; // 3GB

if model_size > MEMORY64_THRESHOLD {
    // Use Memory64 with on-demand layer loading
} else {
    // Use standard in-memory loading
}
```

## Performance Characteristics

### Benchmarks (Measured on Llama-2-7B Q4_K_M, 4.08GB)

| Metric | Standard Loading | Memory64 |
|--------|-----------------|----------|
| **Loading Time** | N/A (out of memory) | 0.01s |
| **Memory Usage** | >4GB | **3.6 MB** |
| **Layer Access** | Instant (in RAM) | 342ms/layer |
| **Total Memory** | Model size | Metadata only |

### Trade-offs

**Advantages:**
- ✅ Load models >4GB that wouldn't fit in memory
- ✅ Minimal memory footprint (only cache + metadata)
- ✅ Fast initial loading (metadata only)
- ✅ LRU cache for frequently accessed layers

**Limitations:**
- ⚠️ Slower layer access (disk I/O vs RAM)
- ⚠️ First inference slower due to cold cache
- ⚠️ Requires native Wasmtime/Wasmer runtime

## Usage Examples

### Example 1: Automatic Loading

```rust
use wasm_chord_runtime::memory64_gguf::Memory64GGUFLoader;
use std::fs::File;

// Load any model - Memory64 activates automatically for >3GB
let file = File::open("model.gguf")?;
let reader = BufReader::new(file);
let mut parser = GGUFParser::new(reader);
let mut loader = Memory64GGUFLoader::new();

let model = loader.load_model(&mut parser)?;
// Model is ready! Memory64 used if model >3GB
```

### Example 2: Manual Configuration

```rust
use wasm_chord_runtime::memory64::{Memory64Runtime, MemoryLayout};

// Create custom Memory64 layout for 30B+ models
let layout = MemoryLayout::multi(&[
    ("embeddings", 2),   // 2GB
    ("layers_0_15", 8),  // 8GB
    ("layers_16_31", 8), // 8GB
    ("lm_head", 2),      // 2GB
])?;

let runtime = Arc::new(Memory64Runtime::new(layout, true));
```

### Example 3: Layer Access with Caching

```rust
// Access layers - automatically cached
let layer = model.get_layer(0)?;  // Cold: 357ms
let layer = model.get_layer(0)?;  // Cached: <1ms

// Check cache statistics
let stats = model.cache_stats();
println!("Hit rate: {:.1}%",
    stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0
);
```

### Example 4: Preloading for Performance

```rust
// Preload all layers before inference (optional)
model.preload_all_layers()?;

// Now all layers are cached for fast access
```

## FFI Functions Reference

### Host-side (Exposed by `memory64-host`)

```rust
// Load layer weights from Memory64 into WASM memory
fn memory64_load_layer(layer_id: u32, wasm_ptr: u32, max_size: u32) -> i32;

// Read arbitrary data from Memory64
fn memory64_read(offset: u64, wasm_ptr: u32, size: u32) -> i32;

// Check if Memory64 is enabled
fn memory64_is_enabled() -> i32;

// Get Memory64 statistics
fn memory64_stats() -> i64;
```

### WASM-side (Imported by `memory64-wasm`)

```rust
// Safe wrapper for layer loading
pub fn load_layer(layer_id: u32, buffer: &mut [u8]) -> Result<usize>;

// Safe wrapper for reading
pub fn read(offset: u64, buffer: &mut [u8]) -> Result<usize>;

// Check availability
pub fn is_enabled() -> bool;
```

## Configuration Options

### Memory Layout Strategies

#### Single Region (7B-13B models)
```rust
let layout = MemoryLayout::single(8, "model_storage")?; // 8GB
```

#### Multi-Region (30B+ models)
```rust
let layout = MemoryLayout::multi(&[
    ("embeddings", 4),
    ("layers_part1", 10),
    ("layers_part2", 10),
    ("lm_head", 4),
])?;
```

### Cache Configuration

```rust
// Adjust cache size based on available RAM
let layer_manager = Memory64LayerManager::new(
    runtime,
    config,
    8,  // Cache up to 8 layers (adjust based on RAM)
);
```

**Cache Size Guidelines:**
- 4 layers: ~800MB RAM (minimum)
- 8 layers: ~1.6GB RAM (recommended)
- 16 layers: ~3.2GB RAM (high performance)

## Integration Patterns

### Pattern 1: Browser + Native Backend

```typescript
// Browser (standard loading)
import { WasmModel } from 'wasm-chord';

const model = new WasmModel(modelBytes);  // <3.5GB only
```

```rust
// Native server (Memory64 for large models)
use wasm_chord_runtime::memory64_gguf::Memory64GGUFLoader;

let model = loader.load_model(&mut parser)?;  // Any size
```

### Pattern 2: Node.js with Native Addon

```javascript
// Load via native addon (uses Memory64)
const { loadModel } = require('./native');

const model = await loadModel('llama-2-7b.gguf');
// Memory64 automatically enabled
```

### Pattern 3: Desktop Application (Tauri)

```rust
// Tauri backend command
#[tauri::command]
async fn load_large_model(path: String) -> Result<String, String> {
    let model = Memory64GGUFLoader::new()
        .load_model_from_path(&path)?;
    Ok("Model loaded with Memory64".to_string())
}
```

## Troubleshooting

### Issue: "Model too large for browser WASM"

**Solution:** Use native runtime with Memory64, or quantize model to <3.5GB

### Issue: Slow inference speed

**Solutions:**
1. Increase cache size: `Memory64LayerManager::new(runtime, config, 16)`
2. Preload layers: `model.preload_all_layers()?`
3. Use faster storage (NVMe SSD)

### Issue: "Memory64 not available"

**Cause:** Running in browser or WASM environment without Wasmtime

**Solution:**
- For browsers: Use smaller models with standard loading
- For native: Ensure Wasmtime is properly configured

### Issue: High memory usage

**Cause:** Cache too large or memory leak

**Solutions:**
1. Reduce cache size: `Memory64LayerManager::new(runtime, config, 4)`
2. Check cache stats: `model.cache_stats()`
3. Manually evict: `model.clear_cache()`

## API Documentation

### `Memory64Runtime`

Core runtime managing Memory64 regions.

```rust
impl Memory64Runtime {
    pub fn new(layout: MemoryLayout, enable_logging: bool) -> Self;
    pub fn add_to_linker(&self, linker: &mut Linker<()>) -> Result<()>;
    pub fn read(&self, offset: u64, buffer: &mut [u8]) -> Result<usize>;
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()>;
}
```

### `Memory64LayerManager`

Manages layer caching with LRU eviction.

```rust
impl Memory64LayerManager {
    pub fn new(runtime: Arc<Memory64Runtime>, config: TransformerConfig, max_cached: usize) -> Self;
    pub fn load_layer(&mut self, layer_id: u32) -> Result<Arc<LayerWeights>>;
    pub fn cache_stats(&self) -> CacheStats;
}
```

### `Memory64Model`

High-level model interface with on-demand loading.

```rust
impl Memory64Model {
    pub fn new(/* ... */) -> Self;
    pub fn get_layer(&mut self, layer_id: u32) -> Result<Arc<LayerWeights>>;
    pub fn preload_all_layers(&mut self) -> Result<()>;
    pub fn cache_stats(&self) -> CacheStats;
    pub fn clear_cache(&mut self);
}
```

### `Memory64GGUFLoader`

GGUF model loader with automatic Memory64 detection.

```rust
impl Memory64GGUFLoader {
    pub fn new() -> Self;
    pub fn load_model<R: Read + Seek>(&mut self, parser: &mut GGUFParser<R>) -> Result<Memory64Model>;
    pub fn load_tensor_lazy<R: Read + Seek>(&mut self, tensor_name: &str, parser: &mut GGUFParser<R>) -> Result<Vec<f32>>;
}
```

## Best Practices

### 1. Choose the Right Loading Strategy

```rust
// Let wasm-chord decide automatically
let model = Memory64GGUFLoader::new().load_model(&mut parser)?;

// Model <3GB: Standard loading (fast, more RAM)
// Model >3GB: Memory64 (slower, minimal RAM)
```

### 2. Optimize Cache Size

```rust
// Calculate based on available RAM
let available_ram_gb = 8.0;  // 8GB available
let cache_size = (available_ram_gb / 0.2) as usize;  // ~200MB per layer

let manager = Memory64LayerManager::new(runtime, config, cache_size);
```

### 3. Monitor Performance

```rust
let stats = model.cache_stats();
if stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 < 0.5 {
    println!("⚠️  Low cache hit rate - consider increasing cache size");
}
```

### 4. Preload for Interactive Use

```rust
// For chatbots and interactive apps
println!("Warming up model...");
model.preload_all_layers()?;
println!("Ready for inference!");
```

## Roadmap

### Phase 1 (Complete ✅)
- Memory64 runtime with Wasmtime
- FFI bridge for WASM access
- On-demand layer loading
- LRU cache management
- GGUF integration

### Phase 2 (Future)
- Multi-threaded layer loading
- GPU integration with Memory64
- Persistent cache (mmap)
- Optimized quantization decompression
- Browser Memory64 support (when available)

### Phase 3 (Research)
- Distributed inference across multiple runtimes
- Streaming layer loading from network
- Dynamic cache eviction strategies
- Memory64 + WebGPU integration

## License

MIT OR Apache-2.0
