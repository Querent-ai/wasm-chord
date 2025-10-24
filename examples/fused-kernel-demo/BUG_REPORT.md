# Q4_K Dequantization NaN Bug

## Issue
When dequantizing Q4_K blocks from GGUF files, approximately 4% of blocks produce NaN values.

## Symptoms
- 451,328 NaN values out of 11,534,336 total elements (3.9%)
- 256 Inf values
- Affects blocks 11, 20, 49, 72, 98, etc. (1764 blocks total out of 45056)
- These blocks have f16 NaN values in the `d` or `dmin` scale fields

## Investigation Results

### ✅ What's Working
- Block structure is correct (matches llama.cpp: 144 bytes, correct field order)
- Dequantization algorithm is correct (matches llama.cpp reference implementation)
- Block 0 dequantizes successfully without NaN
- The model file itself is valid (llama-cli works perfectly)

### ❌ What's Broken
- Reading blocks from the GGUF file produces corrupt data for ~4% of blocks
- Example bad block (#11):
  - d = 0xfe09 (f16 NaN) 
  - dmin = 0x6550 (valid)
  - All 256 dequantized values are NaN

### Root Cause Hypothesis
The tensor data is being read from incorrect byte offsets, causing block boundaries to be misaligned. This results in random file data being interpreted as the `d`/`dmin` f16 scale values, some of which happen to be NaN encodings.

## Reproduction
```bash
cd /home/puneet/wasm-chord/examples/fused-kernel-demo
cargo run --release --bin check_all_blocks
```

## Model Info
- File: /home/puneet/.ollama/models/tinyllama-1.1b.Q4_K_M.gguf
- Tensor: blk.0.ffn_gate.weight
- Shape: [2048, 5632]
- Type: Q4_K
- Size: 6,488,064 bytes (45,056 blocks × 144 bytes)

## Next Steps
1. Verify tensor offset calculation in GGUFParser
2. Check for alignment issues in block reading
3. Compare byte-by-byte with llama.cpp's tensor loading
4. Consider using llama.cpp's GGUF parser directly via FFI
