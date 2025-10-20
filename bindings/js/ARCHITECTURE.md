# Package Architecture Overview

## 🎯 **Two-Tier Architecture**

wasm-chord uses a **hybrid architecture** optimized for different deployment scenarios:

### 1. **Browser Package** (`@querent/wasm-chord-web`)
- **Target**: Web browsers with WebGPU
- **Memory**: Standard WASM memory (<4GB)
- **Models**: Small models (TinyLlama, small Llamas)
- **Features**: WebGPU acceleration, streaming, chat templates
- **Compilation**: Pure WASM (no host dependencies)

### 2. **Node.js Package** (`@querent/wasm-chord-node`)  
- **Target**: Node.js servers and desktop apps
- **Memory**: Host memory with Memory64 bridge
- **Models**: Large models (7B, 13B, 70B+)
- **Features**: Memory64, CUDA, on-demand loading
- **Compilation**: WASM + host runtime (Wasmtime)

## 🔧 **Technical Details**

### Browser Limitations
```typescript
// ❌ This won't work in browsers
const model = await runtime.loadModel('llama-2-7b.gguf'); // 4GB model
// Error: "Model too large for browser WASM memory"

// ✅ This works in browsers  
const model = await runtime.loadModel('tinyllama.gguf'); // 0.67GB model
```

### Node.js Capabilities
```typescript
// ✅ This works in Node.js
const model = await runtime.loadModel('llama-2-7b.gguf'); // 4GB model
// Automatically uses Memory64 for on-demand loading
```

## 📦 **Package Contents**

### Web Package (`@querent/wasm-chord-web`)
```
pkg/
├── wasm_chord_runtime.js      # WASM bindings
├── wasm_chord_runtime_bg.wasm # Compiled WASM
└── wasm_chord_runtime.d.ts    # TypeScript definitions
```

### Node Package (`@querent/wasm-chord-node`)
```
pkg/
├── wasm_chord_runtime.js      # WASM bindings (standard)
├── wasm_chord_runtime_bg.wasm # Compiled WASM (standard)
└── wasm_chord_runtime.d.ts    # TypeScript definitions
```

**Note**: Memory64 features are provided by the **host runtime** (Wasmtime), not the WASM module itself.

## 🚀 **Deployment Strategy**

### Small Models (<4GB)
- **Web**: Use `@querent/wasm-chord-web` in browsers
- **Node**: Use `@querent/wasm-chord-node` (standard loading)

### Large Models (>4GB)  
- **Web**: Not supported (browser limitation)
- **Node**: Use `@querent/wasm-chord-node` (Memory64 host runtime)

## 🔄 **Migration Path**

### From Browser to Node.js
```typescript
// Browser code
import { WasmChordBrowser } from '@querent/wasm-chord-web';
const runtime = await WasmChordBrowser.init();

// Node.js code (same API!)
import { WasmChordNode } from '@querent/wasm-chord-node';
const runtime = await WasmChordNode.init();
```

### Feature Detection
```typescript
// Check capabilities
const capabilities = await getBrowserCapabilities();
if (capabilities.webgpu) {
  console.log('WebGPU available');
}

const nodeCapabilities = await getNodeCapabilities();
if (nodeCapabilities.memory64) {
  console.log('Memory64 available');
}
```

## 📊 **Performance Comparison**

| Scenario | Browser | Node.js |
|----------|---------|---------|
| TinyLlama (0.67GB) | 2-5 tok/s (WebGPU) | 3-7 tok/s (CPU) |
| Llama-2-7B (4GB) | ❌ Not supported | 1-3 tok/s (Memory64) |
| Memory Usage | ~800MB | ~200MB (Memory64) |
| Loading Time | ~2s | ~0.01s (Memory64) |

This architecture ensures optimal performance for each deployment scenario while maintaining API compatibility.
