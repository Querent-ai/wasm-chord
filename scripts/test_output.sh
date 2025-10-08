#!/bin/bash
cd /home/puneet/wasm-chord/examples/simple-generation
timeout 30 cargo run 2>&1 | grep -E "(Assistant response|📝|🎯 Top 5 logits|🎲 Sampled token)" | tail -20

