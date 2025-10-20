# @querent/wasm-chord-node

WebAssembly LLM inference runtime optimized for Node.js with Memory64 support for large models.

## Features

- 🧠 **Memory64 Support** - Handle models >4GB with on-demand loading
- ⚡ **CUDA Acceleration** - GPU-accelerated inference (optional)
- 📦 **Large Models** - Support for 7B, 13B, 70B+ models
- 🔄 **Lazy Loading** - Load layers only when needed
- 📈 **Prefetching** - Pre-load next layers for speed
- 🎯 **Chat Templates** - Built-in chat formatting
- 🔧 **Production Ready** - Optimized for server deployment

## Installation

```bash
npm install @querent/wasm-chord-node
```

## Quick Start

```typescript
import { WasmChordNode } from '@querent/wasm-chord-node';

// Initialize runtime
const runtime = await WasmChordNode.init({
  cudaEnabled: false,  // Enable CUDA if available
  maxMemoryBytes: 16_000_000_000,  // 16GB for large models
  prefetchDistance: 2  // Pre-load next 2 layers
});

// Load model (automatically uses Memory64 for large models)
const model = await runtime.loadModel('./models/llama-2-7b-chat-q4_k_m.gguf');

// Generate text
const response = await model.generate("Hello, how are you?");
console.log(response);

// Or stream tokens
for await (const token of model.generateStream("Tell me a story")) {
  process.stdout.write(token);
}
```

## Memory64 Features

### Automatic Detection
- Models <3GB: Standard loading (all weights in RAM)
- Models >3GB: Memory64 activation (on-demand layer loading)

### Performance Benefits
- **Memory Usage**: 99.9% reduction (4GB model → 3.6MB RAM)
- **Loading Time**: 0.01s (metadata only)
- **Layer Access**: 342ms per layer (cold), ~50ms (warm)

### Configuration
```typescript
const model = await runtime.loadModel('./model.gguf', {
  prefetchDistance: 3  // Pre-load next 3 layers
});

// Adjust prefetch distance at runtime
model.setPrefetchDistance(5);
```

## Model Support

### Supported Sizes
- **Small**: <3GB (standard loading)
- **Large**: 3GB-100GB+ (Memory64)
- **Examples**: TinyLlama (0.67GB), Llama-2-7B (4.08GB), Llama-2-70B (40GB+)

### Supported Formats
- GGUF v2/v3 (quantized models)
- Q4_K_M, Q8_0, F16 quantization

## API Reference

### WasmChordNode

```typescript
class WasmChordNode {
  static async init(config?: NodeInitConfig): Promise<WasmChordNode>
  async loadModel(source: string | Buffer, opts?: LoadOptions): Promise<NodeModel>
  async hasCUDA(): Promise<boolean>
  getRuntimeInfo(): RuntimeInfo
}
```

### NodeModel

```typescript
class NodeModel {
  getModelInfo(): NodeModelInfo
  async generate(prompt: string, opts?: NodeGenOptions): Promise<string>
  async *generateStream(prompt: string, opts?: NodeGenOptions): AsyncIterable<string>
  setPrefetchDistance(distance: number): void
  getMemory64Stats(): Memory64Stats | null
  async free(): Promise<void>
}
```

## Configuration

```typescript
interface NodeInitConfig {
  maxMemoryBytes?: number;     // Default: 16GB
  deterministic?: boolean;     // Default: false
  cudaEnabled?: boolean;       // Default: false
  numThreads?: number;        // Default: os.cpus().length
  prefetchDistance?: number;  // Default: 1
}
```

## Examples

### Basic Generation

```typescript
const runtime = await WasmChordNode.init();
const model = await runtime.loadModel('./models/llama-2-7b-chat-q4_k_m.gguf');
const text = await model.generate("The capital of France is");
console.log(text); // "Paris"
```

### Memory64 Statistics

```typescript
const model = await runtime.loadModel('./models/large-model.gguf');
const info = model.getModelInfo();

console.log('Memory64 Active:', info.memory64Active);
console.log('Model Size:', (info.sizeBytes / 1_000_000_000).toFixed(2), 'GB');

if (info.memory64Stats) {
  console.log('Cached Layers:', info.memory64Stats.cachedLayers);
  console.log('Cache Size:', info.memory64Stats.cacheSize, 'MB');
}
```

### Streaming with Progress

```typescript
const model = await runtime.loadModel('./models/llama-2-7b-chat-q4_k_m.gguf');
let tokenCount = 0;

for await (const token of model.generateStream("Write a short story")) {
  process.stdout.write(token);
  tokenCount++;
  
  if (tokenCount % 10 === 0) {
    const stats = model.getMemory64Stats();
    console.log(`\n[${tokenCount} tokens] Cache: ${stats?.cachedLayers}/${stats?.totalLayers} layers`);
  }
}
```

### CUDA Acceleration

```typescript
const runtime = await WasmChordNode.init({
  cudaEnabled: true
});

if (await runtime.hasCUDA()) {
  console.log('🚀 CUDA acceleration enabled');
} else {
  console.log('💻 Using CPU backend');
}
```

## Performance Optimization

### Prefetch Distance
- **Cold Start**: Higher prefetch (3-5) for better throughput
- **Warm Cache**: Lower prefetch (1-2) for lower memory usage
- **Memory Constrained**: Set to 0 to disable prefetching

### Memory Management
```typescript
// Monitor memory usage
const stats = model.getMemory64Stats();
console.log(`Memory: ${stats?.cacheSize}MB, Layers: ${stats?.cachedLayers}/${stats?.totalLayers}`);

// Adjust prefetch based on available memory
if (stats && stats.cacheSize > 1000) { // >1GB cache
  model.setPrefetchDistance(1); // Reduce prefetch
}
```

## Troubleshooting

### CUDA Issues
```typescript
const capabilities = await getNodeCapabilities();
if (!capabilities.cuda) {
  console.log('CUDA not available, check NVIDIA drivers');
}
```

### Memory Issues
```typescript
// Check if Memory64 is active
const info = model.getModelInfo();
if (!info.memory64Active) {
  console.log('Large model detected but Memory64 not active');
}
```

### Performance Issues
```typescript
// Monitor layer loading times
const stats = model.getMemory64Stats();
console.log(`Cache hit rate: ${stats?.cachedLayers}/${stats?.totalLayers}`);
```

## License

MIT OR Apache-2.0
