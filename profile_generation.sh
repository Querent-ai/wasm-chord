#!/bin/bash
# Profile one token generation

export PROFILE=1
timeout 30 cargo run --release --manifest-path examples/simple-generation/Cargo.toml 2>&1
