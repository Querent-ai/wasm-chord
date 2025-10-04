# 🎵 wasm-chord - Quick Start

## 1. First Build (30 seconds)

```bash
cd wasm-chord

# Install Rust wasm target (one-time setup)
rustup target add wasm32-unknown-unknown

# Build everything
make build

# Run tests
make test
```

**Expected output:** ✅ All 17 tests passing

---

## 2. Try the Web Demo

```bash
# Start local server
make demo

# Open browser to http://localhost:8000
```

You'll see a beautiful UI for:
- Model loading (placeholder)
- Token-by-token streaming
- Parameter controls (temperature, max tokens)

---

## 3. Development Workflow

```bash
# Before committing
make ci-local

# This runs:
# - Format check
# - Clippy lints
# - cargo-deny
# - All tests
# - Wasm build
```

---

## 4. Common Commands

```bash
# Format code
make format

# Fix lints automatically
make lint-fix

# Generate documentation
make docs-open

# Clean build
make clean && make build
```

---

## 5. Project Structure

```
wasm-chord/
├── crates/
│   ├── wasm-chord-core/       ← Tensor ops, GGUF parser
│   ├── wasm-chord-cpu/        ← CPU kernels
│   ├── wasm-chord-gpu/        ← GPU backend (Phase 2)
│   └── wasm-chord-runtime/    ← C ABI exports
├── bindings/js/               ← TypeScript API
├── examples/web-demo/         ← Browser demo
└── Makefile                   ← Build commands
```

---

## 6. Next Steps

1. Read the [full README](README.md)
2. Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture
3. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
4. Review [docs/SETUP.md](docs/SETUP.md) for detailed setup

---

## 7. Phase 2 (Next)

Want to contribute? We need help with:

- [ ] WebGPU compute shaders (WGSL)
- [ ] SIMD optimizations (AVX2, NEON)
- [ ] Tokenizer integration
- [ ] Model caching (IndexedDB)
- [ ] Documentation and examples

---

## 8. Get Help

```bash
# See all available commands
make help

# Run specific crate tests
make test-core
make test-cpu

# Check what CI will run
make ci-local
```

---

## 🎯 Success Checklist

- [x] `make build` completes without errors
- [x] `make test` shows 17 passing tests
- [x] `make demo` opens web UI
- [x] `make ci-local` passes all checks

**You're ready to develop! 🚀**
