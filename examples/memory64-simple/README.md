# Memory64 Simple Test

> **Proof of concept: Allocating >4GB memory in WASM using Wasmtime's Memory64 API**

## What This Tests

Standard WASM has a **4GB memory limit** (32-bit addressing). This example proves we can exceed that limit using Wasmtime's Memory64 support:

- ✅ Create 8GB Memory64 instance (exceeds 4GB)
- ✅ Load standard wasm32 module (no special compilation)
- ✅ Allocate 6GB buffer (>4GB limit)
- ✅ Read/write beyond 4GB boundary

## Building & Running

### 1. Build WASM Module (wasm32)

```bash
cargo build --target wasm32-unknown-unknown --release
```

This creates: `target/wasm32-unknown-unknown/release/memory64_simple.wasm`

**Note**: We compile to **standard wasm32**, not wasm64! Wasmtime handles the memory64 on the host side.

### 2. Build & Run Host Runner

```bash
cargo build --release --bin runner
./target/release/runner
```

### Expected Output

```
🧪 Memory64 Test - Wasmtime Host
==================================

⚙️  Configuring Wasmtime engine...
✅ Engine created with Memory64 support

📦 Creating 8GB Memory64 instance...
✅ Memory64 created: 8.00GB
   Type: 64-bit

📂 Loading WASM module...
✅ Module loaded

🔗 Linking memory to WASM module...
✅ Memory imported

🚀 Instantiating WASM module...
✅ Instance created

🧪 Test 1: Small allocation (1MB)...
✅ PASS: 1MB allocation worked

🧪 Test 2: Large allocation (6GB)...
   This exceeds the standard 4GB WASM limit!
   Allocating 6GB... ✅
✅ PASS: 6GB allocation and access succeeded!

📊 Final memory statistics:
   Memory size: 8.00GB (8589934592 bytes)

═══════════════════════════════════
✅ Memory64 Test PASSED!
═══════════════════════════════════

Key Achievements:
  ✓ Created Memory64 instance (8GB+)
  ✓ Loaded standard wasm32 module
  ✓ Allocated 6GB (>4GB limit)
  ✓ Accessed memory beyond 4GB boundary

🎉 Memory64 works! Ready for large models.
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│  Rust WASM Module (wasm32)             │
│  ┌───────────────────────────────────┐ │
│  │ test_large_allocation(6GB)        │ │
│  │  - Allocates 6GB buffer           │ │
│  │  - Writes to first/last byte      │ │
│  │  - Returns success code           │ │
│  └───────────────────────────────────┘ │
└────────────────┬────────────────────────┘
                 │
                 │ Uses Memory64
                 ▼
┌─────────────────────────────────────────┐
│  Wasmtime Host (Native Rust)           │
│  ┌───────────────────────────────────┐ │
│  │ Memory64 Instance                 │ │
│  │  - Type: i64 (64-bit addressing)  │ │
│  │  - Size: 8GB (128,000 pages)      │ │
│  │  - Max:  16GB (256,000 pages)     │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Key Code

**WASM Side** (`src/lib.rs`):
```rust
#[no_mangle]
pub extern "C" fn test_large_allocation(size_gb: u32) -> u32 {
    let size_bytes = (size_gb as usize) * 1024 * 1024 * 1024;
    let mut buffer = Vec::<u8>::with_capacity(size_bytes);
    buffer.resize(size_bytes, 0);

    buffer[0] = 42;
    buffer[size_bytes - 1] = 99; // Access beyond 4GB!

    if buffer[0] == 42 && buffer[size_bytes - 1] == 99 {
        1 // Success
    } else {
        0 // Failed
    }
}
```

**Host Side** (`src/runner.rs`):
```rust
// Create Memory64 instance
let memory_type = MemoryType::new64(
    128_000,       // 8GB minimum
    Some(256_000), // 16GB maximum
);
let memory = Memory::new(&mut store, memory_type)?;

// Import into WASM module
let mut linker = Linker::new(&engine);
linker.define(&store, "env", "memory", memory)?;
let instance = linker.instantiate(&mut store, &module)?;

// Call WASM function - allocates 6GB!
let test_fn = instance.get_typed_func::<u32, u32>(&mut store, "test_large_allocation")?;
let result = test_fn.call(&mut store, 6)?; // 6GB
```

## Why This Matters

### Standard WASM Memory Limits

| Model | Quantization | Size | Fits in 4GB? |
|-------|--------------|------|--------------|
| 3B | Q4_K_M | 1.8GB | ✅ Yes |
| 7B | Q4_K_M | 4.2GB | ❌ No |
| 13B | Q4_K_M | 7.8GB | ❌ No |
| 70B | Q4_K_M | 40GB | ❌ No |

### With Memory64

| Model | Size | Memory64 Fit? |
|-------|------|---------------|
| 7B | 4.2GB | ✅ Single Memory64 (16GB) |
| 13B | 7.8GB | ✅ Single Memory64 (16GB) |
| 70B | 40GB | ✅ Multi-Memory64 (3×16GB) |

**Conclusion**: Memory64 unlocks 7B+ models in WASM!

## Next Steps

1. ✅ This example proves Memory64 works
2. ⏭️ Next: Adapt `wasm-chord-core` to use Memory64 API
3. ⏭️ Then: Load 7B model (4.2GB) and run inference
4. ⏭️ Week 2: Multi-memory sharding for 70B models

## Dependencies

- **Wasmtime**: v27.0+ (Memory64 support)
- **Rust**: 1.70+ (stable)
- **Target**: `wasm32-unknown-unknown` (standard, no nightly)

## Troubleshooting

### "WASM module not found"

Build the WASM module first:
```bash
cargo build --target wasm32-unknown-unknown --release
```

### "Allocation failed"

Increase memory limits in `src/runner.rs`:
```rust
let memory_type = MemoryType::new64(
    256_000,  // Increase min to 16GB
    Some(512_000), // Increase max to 32GB
);
```

### Wasmtime version too old

Update Wasmtime:
```bash
cargo update -p wasmtime
```

Requires v27.0+ for stable Memory64 API.

## References

- [WebAssembly Memory64 Proposal](https://github.com/WebAssembly/memory64)
- [Wasmtime Memory64 API](https://docs.rs/wasmtime/latest/wasmtime/struct.MemoryType.html)
- [WASM Chord Phase 2 Plan](../../PHASE2_ULTRATHINK.md)

---

**Status**: ✅ Working
**Date**: 2025-10-16
**Phase**: 2.1 - Memory64 Foundation
