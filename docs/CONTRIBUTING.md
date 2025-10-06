# Contributing to wasm-chord

Thank you for your interest in contributing to wasm-chord! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/wasm-chord.git
   cd wasm-chord
   ```
3. **Install dependencies**:
   - Rust 1.75+ with `wasm32-unknown-unknown` target
   - `wasm-pack` for building WebAssembly modules
   - `cargo-deny` for license and security checks (optional but recommended)

4. **Build the project**:
   ```bash
   make build
   ```

## Development Workflow

### Before Submitting a PR

Run these commands to ensure your changes meet our standards:

```bash
# Format code
make format

# Run all checks
make ci-local
```

This will run:
- Code formatting check
- Clippy lints
- Cargo deny checks (licenses, advisories)
- All tests
- WebAssembly build

### Code Style

- Follow Rust naming conventions
- Use `rustfmt` for formatting (enforced in CI)
- Run `make lint` before committing
- Write doc comments for public APIs
- Add tests for new functionality

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issues when applicable (e.g., "Fix #123")

Example:
```
Add SIMD optimization for Q4 dequantization

- Implement AVX2 path for x86_64
- Add fallback for non-SIMD platforms
- Benchmark shows 3x speedup

Fixes #42
```

## Testing

### Running Tests

```bash
# All tests
make test

# Specific crate
make test-core
make test-cpu

# With output
cargo test -- --nocapture
```

### Writing Tests

- Add unit tests in the same file as the code being tested
- Use `#[cfg(test)]` modules
- Test edge cases and error conditions

## Documentation

- Document all public APIs with doc comments
- Include examples in doc comments when helpful
- Update README.md for user-facing changes
- Run `make docs` to verify documentation builds

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub

5. **Address review feedback** if any

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass (`make test`)
- [ ] CI checks pass (`make ci-local`)
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

## Areas for Contribution

We especially welcome contributions in these areas:

- **WebGPU backend**: Implementing compute shaders and kernels
- **SIMD optimizations**: CPU kernel performance improvements
- **Model formats**: Support for ONNX, SafeTensors
- **Documentation**: Examples, tutorials, guides
- **Testing**: More comprehensive test coverage
- **Benchmarks**: Performance testing on various platforms

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Join our [Discord](https://discord.gg/querent) (if available)

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).
