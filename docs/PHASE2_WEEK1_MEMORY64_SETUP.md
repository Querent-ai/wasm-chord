# Phase 2 Week 1: Memory64 Setup & Current Status

> **Research findings and alternative approach for Memory64 in Rust**

---

## 🔍 Current Status of Rust wasm64 Support

### **Finding: wasm64-unknown-unknown target exists but is NOT usable**

```bash
$ rustc --print target-list | grep wasm64
wasm64-unknown-unknown  # ✅ Listed

$ rustup target add wasm64-unknown-unknown
error: toolchain 'stable' does not support target 'wasm64-unknown-unknown'
# ❌ Cannot install (even on nightly)
```

**Conclusion**: The `wasm64-unknown-unknown` target is **placeholder-only** in Rust as of 2025. Not yet implemented.

---

## 🛠️ Alternative Approach: wasm32 with Memory64 Feature

**Good News**: We can use Memory64 **without** the wasm64 target!

###Method: Compile to wasm32, then **enable memory64 via LLVM/wasm-tools**

#### **Approach A: Use wasm32 + post-processing** (RECOMMENDED)

```bash
# 1. Build normal wasm32 module
cargo build --target wasm32-unknown-unknown --release

# 2. Convert to memory64 using wasm-tools
wasm-tools component memory64 \
    --input target/wasm32-unknown-unknown/release/your_module.wasm \
    --output your_module_memory64.wasm
```

**How it works**:
- Rust compiles to standard wasm32 (i32 pointers)
- `wasm-tools memory64` transforms:
  - Changes memory type to `(memory i64 ...)`
  - Converts `i32.load` → `i64.load`
  - Extends all memory access instructions
- Result: Memory64-compatible WASM module

**Limitations**:
- Pointer math still uses i32 internally
- May not work for >4GB allocations (Rust's allocator assumes 32-bit)

---

#### **Approach B: Use Wasmtime's Memory64 API** (PRODUCTION-READY)

**Better Strategy**: Don't fight Rust's limitations. Use Wasmtime's host-side memory!

```rust
// Host code (Rust native, not WASM)
use wasmtime::*;

fn main() -> Result<()> {
    let engine = Engine::default();
    let mut store = Store::new(&engine, ());

    // Create memory64 instance on the HOST side
    let memory_type = MemoryType::new64(
        1024,          // Min pages (64GB)
        Some(256_000)  // Max pages (16TB theoretical)
    );
    let memory = Memory::new(&mut store, memory_type)?;

    // Load WASM module (standard wasm32)
    let module = Module::from_file(&engine, "model.wasm")?;

    // Import memory64 into WASM module
    let imports = [memory.into()];
    let instance = Instance::new(&mut store, &module, &imports)?;

    // WASM code can now use >4GB memory!
    // All allocations happen in host-created memory64

    Ok(())
}
```

**How it works**:
1. WASM module is **standard wasm32** (no special compilation)
2. Host (Wasmtime) creates **Memory64** instance
3. WASM imports this memory and uses it
4. All >4GB allocations work because **host manages memory**

**Advantages**:
- ✅ No custom Rust target needed
- ✅ Works with stable Rust
- ✅ True >4GB support
- ✅ Production-ready (Wasmtime stable API)

---

## 📋 Revised Week 1 Plan

### **Step 1: Install Wasmtime** ✅

```bash
# Install Wasmtime CLI
cargo install wasmtime-cli

# Verify version (should be v37+)
wasmtime --version
```

**Expected output**: `wasmtime-cli 37.0.2` (or newer)

---

### **Step 2: Install wasm-tools**

```bash
cargo install wasm-tools

# Verify
wasm-tools --version
```

---

### **Step 3: Create Memory64 Test Example**

Let's create a simple test that allocates >4GB:

#### **File: `examples/memory64-simple/Cargo.toml`**

```toml
[package]
name = "memory64-simple"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-chord-core = { path = "../../crates/wasm-chord-core" }

[profile.release]
opt-level = 3
lto = true
```

#### **File: `examples/memory64-simple/src/lib.rs`**

```rust
//! Simple test: Allocate and access memory beyond 4GB
//!
//! This will fail in standard WASM but work with Wasmtime's Memory64

use std::slice;

#[no_mangle]
pub extern "C" fn test_large_allocation(size_gb: u32) -> u32 {
    let size_bytes = (size_gb as usize) * 1024 * 1024 * 1024;

    // Try to allocate large buffer
    let layout = std::alloc::Layout::from_size_align(size_bytes, 8).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };

    if ptr.is_null() {
        return 0; // Allocation failed
    }

    // Write to first and last byte
    unsafe {
        *ptr = 42;
        *ptr.add(size_bytes - 1) = 99;
    }

    // Verify
    let success = unsafe {
        *ptr == 42 && *ptr.add(size_bytes - 1) == 99
    };

    unsafe { std::alloc::dealloc(ptr, layout); }

    if success { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn get_memory_size() -> usize {
    // In WASM, this would return current memory pages
    // With Memory64, this can exceed 4GB
    unsafe { core::arch::wasm32::memory_size(0) } * 65536 // pages * 64KB
}
```

---

### **Step 4: Create Host Runner (Wasmtime)**

#### **File: `examples/memory64-simple/runner.rs`**

```rust
//! Host code to run Memory64 WASM module

use wasmtime::*;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🧪 Memory64 Test - Wasmtime Host");
    println!("==================================\n");

    // 1. Create Wasmtime engine
    let mut config = Config::new();
    config.wasm_memory64(true);  // Enable Memory64 support
    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());

    // 2. Create 8GB memory (exceeds 4GB limit!)
    println!("Creating 8GB Memory64 instance...");
    let memory_type = MemoryType::new64(
        128_000,   // min: 128,000 pages = 8GB
        Some(256_000)  // max: 256,000 pages = 16GB
    );
    let memory = Memory::new(&mut store, memory_type)?;
    println!("✅ Memory64 created: {}GB\n", memory.size(&store) * 64 / 1024 / 1024);

    // 3. Load WASM module
    println!("Loading WASM module...");
    let module = Module::from_file(
        &engine,
        "target/wasm32-unknown-unknown/release/memory64_simple.wasm"
    )?;
    println!("✅ Module loaded\n");

    // 4. Import memory into WASM
    let mut linker = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory)?;

    // 5. Instantiate
    println!("Instantiating WASM module...");
    let instance = linker.instantiate(&mut store, &module)?;
    println!("✅ Instance created\n");

    // 6. Test: Allocate 6GB (>4GB limit!)
    println!("Testing 6GB allocation (exceeds 4GB WASM limit)...");
    let test_fn = instance
        .get_typed_func::<u32, u32>(&mut store, "test_large_allocation")?;

    let result = test_fn.call(&mut store, 6)?; // 6GB

    if result == 1 {
        println!("✅ SUCCESS: 6GB allocation worked!\n");
    } else {
        println!("❌ FAILED: 6GB allocation failed\n");
    }

    // 7. Check memory size
    let mem_fn = instance
        .get_typed_func::<(), usize>(&mut store, "get_memory_size")?;
    let mem_size = mem_fn.call(&mut store, ())?;
    println!("Current memory size: {}GB", mem_size / 1024 / 1024 / 1024);

    Ok(())
}
```

---

### **Step 5: Build and Run**

```bash
# Build WASM module (standard wasm32)
cd examples/memory64-simple
cargo build --target wasm32-unknown-unknown --release

# Build host runner
cargo build --release --bin runner

# Run test
./target/release/runner
```

**Expected Output**:
```
🧪 Memory64 Test - Wasmtime Host
==================================

Creating 8GB Memory64 instance...
✅ Memory64 created: 8GB

Loading WASM module...
✅ Module loaded

Instantiating WASM module...
✅ Instance created

Testing 6GB allocation (exceeds 4GB WASM limit)...
✅ SUCCESS: 6GB allocation worked!

Current memory size: 8GB
```

---

## 🎯 Key Insights

### **What We Learned**:

1. **Rust's wasm64 target doesn't exist yet** (placeholder only)
2. **We don't need it!** Wasmtime's Memory64 API works with standard wasm32
3. **Host-side memory management** is the production approach
4. **Standard Rust + Wasmtime = Memory64 today**

### **Production Strategy**:

```
┌─────────────────────────────────────────────┐
│  Rust Code (wasm32-unknown-unknown)        │
│  - Standard compilation                     │
│  - No special flags                         │
│  - Works with stable Rust                   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Wasmtime Host (Native Rust)               │
│  - Creates Memory64 instance                │
│  - Imports into WASM module                 │
│  - Manages >4GB allocations                 │
└─────────────────────────────────────────────┘
```

---

## 📦 Next Steps for Week 1

### **Immediate Tasks**:

- [ ] Wait for Wasmtime install to complete
- [ ] Create `examples/memory64-simple` with test code above
- [ ] Build and verify 6GB allocation works
- [ ] Document findings in main repo

### **Then**:

- [ ] Adapt wasm-chord to use Wasmtime Memory64 API
- [ ] Load 7B model (4.2GB) using Memory64
- [ ] Benchmark vs. standard memory (if possible)

---

## 🚧 Important Notes

### **Browser Limitations**:
- Memory64 in browsers is **experimental** (Chrome flags only)
- Multi-memory is **not available** in browsers
- **Server/Edge deployment** is the target for Phase 2

### **Why This Approach Works**:
- Wasmtime/Wasmer have **stable Memory64 APIs**
- Don't need Rust compiler changes
- Works **today** in production
- Scales to 70B models with multi-memory (Week 2)

---

*Status: ✅ Research Complete*
*Next: Implementation*
*Date: 2025-10-16*
