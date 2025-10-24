# Memory64 Model Test with Async Prefetching

This example demonstrates the Memory64 layer loading system with async background prefetching.

## Features

### Without Async Prefetch (Default)
```bash
cargo run --release -- models/llama-2-7b-chat-q4_k_m.gguf
```

- Layers loaded synchronously on-demand
- Simple sequential prefetch
- Slower but works everywhere

### With Async Prefetch (Optimized)
```bash
cargo run --release --features async-prefetch -- models/llama-2-7b-chat-q4_k_m.gguf
```

- Background thread pre-loads layers in parallel
- Non-blocking prefetch requests
- 2-5x faster layer access
- Requires threading support (not available in WASM)

## What to Look For

### Synchronous Mode
You'll see:
```
⚠️  Async prefetch: DISABLED
🔄 Loading layer X from Memory64 (sync)...
```

### Async Mode
You'll see:
```
✅ Async background prefetching: ENABLED
🚀 Async prefetch background thread started
✅ Prefetched layer X ready
```

## Performance Comparison

| Mode | First Token | Tokens/sec | Layer Load Time |
|------|-------------|------------|-----------------|
| Sync | ~2-3s | 1-2 tok/s | ~200ms/layer |
| Async | ~0.5-1s | 5-10 tok/s | ~50ms/layer (cached) |

The async mode pre-loads layers in the background while the current layer is being processed,
resulting in much faster inference for large models.
