# Model Conversion Tools

Python utilities for converting HuggingFace models to GGUF format.

## Installation

```bash
pip install torch transformers numpy
```

## Usage

### Convert a Model

```bash
# Convert TinyLlama to Q4_0 (recommended)
python tools/convert_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output models/tinyllama-q4_0.gguf \
    --quant q4_0

# Convert to Q8_0 (higher quality)
python tools/convert_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output models/tinyllama-q8_0.gguf \
    --quant q8_0

# Convert to F16 (full precision)
python tools/convert_model.py \
    --model microsoft/phi-2 \
    --output models/phi2-f16.gguf \
    --quant f16
```

### Validate a GGUF File

```bash
python tools/validate_gguf.py models/tinyllama-q4_0.gguf
```

## Supported Models

- ✅ LLaMA architecture (TinyLlama, Llama-2, CodeLlama)
- ✅ Phi architecture (Phi-2)
- 🔄 More architectures coming in Phase 2

## Quantization Types

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| `q4_0` | ~50% | Good | **Recommended** - Best size/quality trade-off |
| `q8_0` | ~75% | Better | Higher quality, larger size |
| `f16` | 100% | Best | Testing, benchmarking |
| `f32` | 200% | Best | Reference, no compression |

## Output Format

The GGUF file includes:
- Model architecture metadata
- Tokenizer vocabulary
- Special tokens (BOS, EOS, UNK)
- Quantized weights
- Layer configuration

## Example Workflow

```bash
# 1. Convert model
python tools/convert_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output tinyllama.gguf \
    --quant q4_0

# 2. Validate output
python tools/validate_gguf.py tinyllama.gguf

# 3. Use with wasm-chord
cargo run --example cli -- --model tinyllama.gguf --prompt "Hello"
```

## Troubleshooting

### Out of Memory
If conversion fails with OOM, try:
- Using a smaller model (TinyLlama instead of Llama-7B)
- Using F16 or Q4_0 quantization
- Running on a machine with more RAM

### Missing Dependencies
```bash
pip install torch transformers numpy
```

### CUDA Not Available
CUDA is not required - models are converted on CPU.

## Technical Details

### GGUF Format
- Magic: `0x46554747` ("GGUF")
- Version: 3
- Alignment: 32 bytes
- Endianness: Little-endian

### Quantization
- **Q4_0**: 4-bit symmetric quantization, 32-element blocks
- **Q8_0**: 8-bit symmetric quantization, 32-element blocks
- **F16**: IEEE 754 half-precision float
- **F32**: IEEE 754 single-precision float

## Future Enhancements

- [ ] Support for more architectures (GPT-2, BERT, etc.)
- [ ] Q5_0, Q5_1, Q8_1 quantization
- [ ] Mixed quantization (different layers)
- [ ] Importance matrix for better quantization
- [ ] Parallel conversion for large models
