SHELL := /bin/bash
.PHONY: all
all: build

## Help display
.PHONY: help
help:
	@echo "🎵 wasm-chord - WebAssembly LLM Inference Runtime"
	@echo ""
	@echo "Available targets:"
	@echo "  make build              - Build all crates (debug)"
	@echo "  make build-release      - Build all crates (release)"
	@echo "  make build-wasm         - Build wasm32 target"
	@echo "  make build-all          - Build all targets"
	@echo ""
	@echo "  make test               - Run all tests"
	@echo "  make test-core          - Run core crate tests"
	@echo "  make test-cpu           - Run CPU backend tests"
	@echo ""
	@echo "  make check              - Run cargo check"
	@echo "  make check-all          - Check all features"
	@echo ""
	@echo "  make lint               - Run format check, clippy, and deny"
	@echo "  make lint-fix           - Auto-fix formatting and clippy issues"
	@echo "  make lint-deny          - Run cargo-deny checks"
	@echo ""
	@echo "  make format             - Format code with rustfmt"
	@echo "  make format-check       - Check formatting without modifying"
	@echo ""
	@echo "  make docs               - Generate documentation"
	@echo "  make docs-open          - Generate and open documentation"
	@echo ""
	@echo "  make wasm-pack          - Build wasm-bindgen package"
	@echo "  make wasm-pack-node     - Build for Node.js target"
	@echo ""
	@echo "  make demo               - Run web demo (requires Python)"
	@echo ""
	@echo "  make clean              - Clean build artifacts"
	@echo "  make clean-all          - Clean everything including caches"

## Build targets
.PHONY: build build-release build-wasm build-all
build:
	cargo build --workspace

build-release:
	cargo build --workspace --release

build-wasm:
	cargo build --target wasm32-unknown-unknown --package wasm-chord-runtime

build-all: build build-release build-wasm

## Test targets
.PHONY: test test-core test-cpu test-runtime test-all
test:
	cargo test --workspace

test-core:
	cargo test --package wasm-chord-core

test-cpu:
	cargo test --package wasm-chord-cpu

test-runtime:
	cargo test --package wasm-chord-runtime

test-all: test

## Check targets
.PHONY: check check-all check-wasm
check:
	SKIP_WASM_BUILD=1 cargo check --workspace

check-all: check check-wasm

check-wasm:
	cargo check --target wasm32-unknown-unknown --package wasm-chord-runtime

## Lint targets
.PHONY: lint lint-fix lint-deny lint-clippy format format-check
lint: format-check lint-clippy

lint-fix:
	cargo fmt --all
	cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged

lint-clippy:
	cargo clippy --workspace --lib -- -D warnings

lint-deny:
	cargo deny check -c deny.toml

format:
	cargo fmt --all

format-check:
	cargo fmt --all -- --check

## Documentation
.PHONY: docs docs-open
docs:
	cargo doc --workspace --no-deps

docs-open:
	cargo doc --workspace --no-deps --open

## Wasm-pack targets
.PHONY: wasm-pack wasm-pack-node wasm-pack-web
wasm-pack:
	cd crates/wasm-chord-runtime && wasm-pack build --target web --out-dir ../../bindings/js/pkg

wasm-pack-node:
	cd crates/wasm-chord-runtime && wasm-pack build --target nodejs --out-dir ../../bindings/js/pkg-node

wasm-pack-web: wasm-pack

## Demo and examples
.PHONY: demo serve-demo
demo: serve-demo

serve-demo:
	@echo "🌐 Starting web demo on http://localhost:8000"
	@echo "   (Press Ctrl+C to stop)"
	cd examples/web-demo && python3 -m http.server 8000

## Clean targets
.PHONY: clean clean-all clean-wasm
clean:
	cargo clean

clean-wasm:
	rm -rf bindings/js/pkg bindings/js/pkg-node

clean-all: clean clean-wasm
	rm -rf target/
	find . -name "Cargo.lock" -type f -delete

## CI-like local checks
.PHONY: ci-local
ci-local: format-check lint-clippy lint-deny test build-wasm
	@echo "✅ All CI checks passed locally!"

## Version management
.PHONY: version
version:
ifndef v
	@echo "Usage: make version v=0.1.0"
	@exit 1
endif
	@echo "Setting version to $(v)"
	@find . -name "Cargo.toml" -type f -not -path "*/target/*" -exec sed -i.bak 's/^version = ".*"/version = "$(v)"/' {} \;
	@find . -name "*.bak" -delete
	@echo "✅ Version updated to $(v)"
