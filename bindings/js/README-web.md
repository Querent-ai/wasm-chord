# @querent/wasm-chord-web

WebAssembly LLM inference runtime optimized for browsers with WebGPU acceleration.

## Features

- 🚀 **WebGPU Acceleration** - GPU-accelerated inference in browsers
- 🌐 **Browser Optimized** - Safe memory limits and streaming support
- 📦 **Small Models** - Optimized for models <4GB (TinyLlama, small Llamas)
- ⚡ **Streaming** - Real-time token generation
- 🎯 **Chat Templates** - Built-in chat formatting
- 🔒 **Memory Safe** - Automatic size validation

## Installation

```bash
npm install @querent/wasm-chord-web
```

## Quick Start

```typescript
import { WasmChordBrowser } from '@querent/wasm-chord-web';

// Initialize runtime
const runtime = await WasmChordBrowser.init({
  webgpuEnabled: true,  // Enable WebGPU acceleration
  maxMemoryBytes: 2_000_000_000  // 2GB safety limit
});

// Load model
const model = await runtime.loadModel('https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin');

// Generate text
const response = await model.generate("Hello, how are you?");
console.log(response);

// Or stream tokens
for await (const token of model.generateStream("Tell me a story")) {
  process.stdout.write(token);
}
```

## Browser Compatibility

- ✅ Chrome 113+ (WebGPU support)
- ✅ Firefox 110+ (WebGPU support)  
- ✅ Safari 16.4+ (WebGPU support)
- ⚠️ Edge (WebGPU experimental)

## Model Size Limits

- **Maximum**: ~4GB (browser WASM memory limit)
- **Recommended**: <2GB for optimal performance
- **Examples**: TinyLlama (0.67GB), small Llamas (1-2GB)

For larger models, use [@querent/wasm-chord-node](./node) instead.

## API Reference

### WasmChordBrowser

```typescript
class WasmChordBrowser {
  static async init(config?: BrowserInitConfig): Promise<WasmChordBrowser>
  async loadModel(source: string | File | ArrayBuffer): Promise<BrowserModel>
  async hasWebGPU(): Promise<boolean>
  getRuntimeInfo(): RuntimeInfo
}
```

### BrowserModel

```typescript
class BrowserModel {
  getModelInfo(): BrowserModelInfo
  async generate(prompt: string, opts?: BrowserGenOptions): Promise<string>
  async *generateStream(prompt: string, opts?: BrowserGenOptions): AsyncIterable<string>
  async free(): Promise<void>
}
```

## Configuration

```typescript
interface BrowserInitConfig {
  maxMemoryBytes?: number;     // Default: 2GB
  deterministic?: boolean;     // Default: false
  webgpuEnabled?: boolean;    // Default: true
  numThreads?: number;         // Default: navigator.hardwareConcurrency
}
```

## Examples

### Basic Generation

```typescript
const runtime = await WasmChordBrowser.init();
const model = await runtime.loadModel('/path/to/model.gguf');
const text = await model.generate("The capital of France is");
console.log(text); // "Paris"
```

### Streaming Generation

```typescript
const model = await runtime.loadModel('/path/to/model.gguf');
for await (const token of model.generateStream("Once upon a time")) {
  document.getElementById('output').textContent += token;
}
```

### File Upload

```typescript
const fileInput = document.getElementById('model-file') as HTMLInputElement;
const file = fileInput.files[0];
const model = await runtime.loadModel(file);
```

## Performance Tips

1. **Enable WebGPU** - 2-5x faster than CPU
2. **Use streaming** - Better UX for long generations
3. **Small models** - Faster loading and inference
4. **Worker threads** - Utilize all CPU cores

## Troubleshooting

### WebGPU Not Available
```typescript
const capabilities = await getBrowserCapabilities();
if (!capabilities.webgpu) {
  console.log('WebGPU not available, using CPU fallback');
}
```

### Model Too Large
```typescript
try {
  const model = await runtime.loadModel(largeModelFile);
} catch (error) {
  console.log('Model too large for browser, use Node.js version');
}
```

## License

MIT OR Apache-2.0
