# wasm-chord Setup Guide

## Quick Start

```bash
# Clone and enter directory
cd wasm-chord

# Check everything builds
make check

# Run tests
make test

# Build release
make build-release

# View all available commands
make help
```

## Project Structure

```
wasm-chord/
├── crates/
│   ├── wasm-chord-core/       # Core tensor primitives, GGUF parser
│   ├── wasm-chord-runtime/    # Wasm runtime with C ABI
│   ├── wasm-chord-cpu/        # CPU SIMD kernels
│   └── wasm-chord-gpu/        # WebGPU backend (Phase 2)
├── bindings/
│   ├── js/                    # TypeScript/JavaScript bindings
│   └── python/                # Python bindings (future)
├── examples/
│   ├── web-demo/              # Browser demo
│   └── cli/                   # Command-line tool
├── wit/                       # WIT interface definitions
├── docs/                      # Documentation
└── Makefile                   # Build automation

## Development

### Prerequisites

- Rust 1.75+ with `wasm32-unknown-unknown` target
- Node.js 20+ (for JS bindings)
- wasm-pack (install with: `cargo install wasm-pack`)
- cargo-deny (optional: `cargo install cargo-deny`)

### Building

```bash
# Standard build
make build

# WebAssembly target
make build-wasm

# All targets
make build-all
```

### Testing

```bash
# All tests
make test

# Specific crate
make test-core
make test-cpu

# With verbose output
cargo test -- --nocapture
```

### Linting

```bash
# Check formatting and clippy
make lint

# Auto-fix issues
make lint-fix

# Check licenses and advisories
make lint-deny
```

### Web Demo

```bash
# Serve web demo on http://localhost:8000
make demo
```

## CI Checks

Run the same checks that CI runs:

```bash
make ci-local
```

This includes:
- Format check
- Clippy
- cargo-deny
- All tests
- Wasm build

## Architecture Overview

### Phase 1 (MVP - Current)
- ✅ Cargo workspace scaffold
- ✅ GGUF streaming parser
- ✅ CPU GEMM kernels
- ✅ C ABI exports
- ✅ JS bindings scaffold
- ✅ Web demo

### Phase 2 (Next)
- [ ] WebGPU backend
- [ ] Token streaming
- [ ] Tokenizer integration
- [ ] Model caching

## Common Commands

```bash
# Format code
make format

# Generate docs
make docs
make docs-open

# Build wasm-pack
make wasm-pack

# Clean build artifacts
make clean
make clean-all

# Set version
make version v=0.2.0
```

## Troubleshooting

### Rust toolchain issues

```bash
rustup update
rustup target add wasm32-unknown-unknown
```

### wasm-pack not found

```bash
cargo install wasm-pack
```

### Tests failing

```bash
# Clean and rebuild
make clean-all
make test
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Next Steps

1. Review the design document in `docs/design.md`
2. Check out the examples in `examples/`
3. Read the API documentation: `make docs-open`
4. Join the development discussion
