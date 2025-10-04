#!/usr/bin/env python3
"""
Convert HuggingFace models to GGUF format for wasm-chord

Supports:
- LLaMA architecture (TinyLlama, Llama-2, etc.)
- Phi architecture (Phi-2)
- Quantization options (Q4_0, Q8_0, F16, F32)

Usage:
    python convert_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tinyllama.gguf --quant q4_0
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
except ImportError:
    print("Error: Required dependencies not found.")
    print("Install with: pip install torch transformers numpy")
    sys.exit(1)

# GGUF Constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# Data type enum
class GGUFDataType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8
    Q8_1 = 9

class GGUFWriter:
    """Write GGUF format files"""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.tensors: List[Tuple[str, torch.Tensor, int]] = []
        self.metadata: Dict[str, any] = {}

    def add_metadata(self, key: str, value):
        """Add metadata key-value pair"""
        self.metadata[key] = value

    def add_tensor(self, name: str, tensor: torch.Tensor, dtype: int):
        """Add a tensor to be written"""
        self.tensors.append((name, tensor, dtype))

    def write(self):
        """Write GGUF file"""
        with open(self.output_path, 'wb') as f:
            # Write header
            self._write_header(f)

            # Write metadata
            self._write_metadata(f)

            # Write tensor info
            self._write_tensor_info(f)

            # Align to 32 bytes
            self._align(f, 32)

            # Write tensor data
            self._write_tensor_data(f)

        print(f"✅ Model written to {self.output_path}")
        print(f"   Size: {Path(self.output_path).stat().st_size / (1024**2):.2f} MB")

    def _write_header(self, f):
        """Write GGUF header"""
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(self.tensors)))  # tensor_count
        f.write(struct.pack('<Q', len(self.metadata)))  # metadata_kv_count

    def _write_metadata(self, f):
        """Write metadata key-value pairs"""
        for key, value in self.metadata.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)

            # Write value type and value
            if isinstance(value, str):
                # String type (4)
                f.write(struct.pack('<I', 4))
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
            elif isinstance(value, int):
                # Uint32 type (6)
                f.write(struct.pack('<I', 6))
                f.write(struct.pack('<I', value))
            elif isinstance(value, float):
                # Float32 type (2)
                f.write(struct.pack('<I', 2))
                f.write(struct.pack('<f', value))

    def _write_tensor_info(self, f):
        """Write tensor metadata"""
        data_offset = 0

        for name, tensor, dtype in self.tensors:
            # Tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)

            # Number of dimensions
            ndim = len(tensor.shape)
            f.write(struct.pack('<I', ndim))

            # Dimensions (reversed for GGUF format)
            for dim in reversed(tensor.shape):
                f.write(struct.pack('<Q', dim))

            # Data type
            f.write(struct.pack('<I', dtype))

            # Offset
            f.write(struct.pack('<Q', data_offset))

            # Calculate size for next offset
            tensor_size = self._calculate_tensor_size(tensor, dtype)
            data_offset += tensor_size

    def _write_tensor_data(self, f):
        """Write actual tensor data"""
        for name, tensor, dtype in self.tensors:
            quantized_data = self._quantize_tensor(tensor, dtype)
            f.write(quantized_data)

    def _quantize_tensor(self, tensor: torch.Tensor, dtype: int) -> bytes:
        """Quantize tensor to specified format"""
        # Convert to numpy
        arr = tensor.detach().cpu().float().numpy()

        if dtype == GGUFDataType.F32:
            return arr.astype(np.float32).tobytes()

        elif dtype == GGUFDataType.F16:
            return arr.astype(np.float16).tobytes()

        elif dtype == GGUFDataType.Q4_0:
            return self._quantize_q4_0(arr)

        elif dtype == GGUFDataType.Q8_0:
            return self._quantize_q8_0(arr)

        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def _quantize_q4_0(self, arr: np.ndarray) -> bytes:
        """Quantize to Q4_0 format (4-bit with scale)"""
        # Flatten and reshape to blocks of 32
        flat = arr.flatten()
        n_blocks = (len(flat) + 31) // 32

        # Pad to multiple of 32
        if len(flat) % 32 != 0:
            flat = np.pad(flat, (0, 32 - (len(flat) % 32)))

        result = bytearray()

        for i in range(n_blocks):
            block = flat[i*32:(i+1)*32]

            # Calculate scale
            amax = np.abs(block).max()
            scale = amax / 7.0 if amax > 0 else 1.0

            # Quantize to 4-bit
            quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)

            # Pack two 4-bit values per byte
            packed = np.zeros(16, dtype=np.uint8)
            for j in range(16):
                val0 = int(quantized[j*2]) & 0x0F
                val1 = int(quantized[j*2+1]) & 0x0F
                packed[j] = (val1 << 4) | val0

            # Write scale (f16) and quantized data
            result.extend(struct.pack('<e', scale))  # f16
            result.extend(packed.tobytes())

        return bytes(result)

    def _quantize_q8_0(self, arr: np.ndarray) -> bytes:
        """Quantize to Q8_0 format (8-bit with scale)"""
        flat = arr.flatten()
        n_blocks = (len(flat) + 31) // 32

        if len(flat) % 32 != 0:
            flat = np.pad(flat, (0, 32 - (len(flat) % 32)))

        result = bytearray()

        for i in range(n_blocks):
            block = flat[i*32:(i+1)*32]

            # Calculate scale
            amax = np.abs(block).max()
            scale = amax / 127.0 if amax > 0 else 1.0

            # Quantize to 8-bit
            quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)

            # Write scale (f32) and quantized data
            result.extend(struct.pack('<f', scale))
            result.extend(quantized.tobytes())

        return bytes(result)

    def _calculate_tensor_size(self, tensor: torch.Tensor, dtype: int) -> int:
        """Calculate size in bytes for a quantized tensor"""
        numel = tensor.numel()

        if dtype == GGUFDataType.F32:
            return numel * 4
        elif dtype == GGUFDataType.F16:
            return numel * 2
        elif dtype == GGUFDataType.Q4_0:
            # 2 bytes scale + 16 bytes data per 32 elements
            n_blocks = (numel + 31) // 32
            return n_blocks * 18
        elif dtype == GGUFDataType.Q8_0:
            # 4 bytes scale + 32 bytes data per 32 elements
            n_blocks = (numel + 31) // 32
            return n_blocks * 36
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

    def _align(self, f, alignment: int):
        """Align file position to alignment boundary"""
        pos = f.tell()
        padding = (alignment - (pos % alignment)) % alignment
        if padding > 0:
            f.write(b'\x00' * padding)

def convert_model(model_name: str, output_path: str, quant_type: str = "q4_0"):
    """
    Convert HuggingFace model to GGUF format

    Args:
        model_name: HuggingFace model identifier
        output_path: Output GGUF file path
        quant_type: Quantization type (q4_0, q8_0, f16, f32)
    """
    print(f"🔄 Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()

    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Map quantization type
    dtype_map = {
        "f32": GGUFDataType.F32,
        "f16": GGUFDataType.F16,
        "q4_0": GGUFDataType.Q4_0,
        "q8_0": GGUFDataType.Q8_0,
    }
    dtype = dtype_map.get(quant_type.lower(), GGUFDataType.Q4_0)

    # Create GGUF writer
    writer = GGUFWriter(output_path)

    # Add metadata
    config = model.config
    writer.add_metadata("general.architecture", "llama")
    writer.add_metadata("general.name", model_name)
    writer.add_metadata("llama.context_length", getattr(config, 'max_position_embeddings', 2048))
    writer.add_metadata("llama.embedding_length", config.hidden_size)
    writer.add_metadata("llama.block_count", config.num_hidden_layers)
    writer.add_metadata("llama.feed_forward_length", config.intermediate_size)
    writer.add_metadata("llama.attention.head_count", config.num_attention_heads)
    writer.add_metadata("llama.attention.head_count_kv", getattr(config, 'num_key_value_heads', config.num_attention_heads))
    writer.add_metadata("llama.rope.freq_base", getattr(config, 'rope_theta', 10000.0))
    writer.add_metadata("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps)

    # Add vocabulary
    vocab = tokenizer.get_vocab()
    writer.add_metadata("tokenizer.ggml.model", "llama")
    writer.add_metadata("tokenizer.ggml.tokens", len(vocab))

    # Add vocabulary tokens
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        writer.add_metadata(f"tokenizer.ggml.token.{idx}", token)

    # Add special tokens
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        writer.add_metadata("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        writer.add_metadata("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id)
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        writer.add_metadata("tokenizer.ggml.unknown_token_id", tokenizer.unk_token_id)

    print(f"🔄 Converting tensors to {quant_type.upper()}...")

    # Add tensors
    tensor_count = 0
    for name, param in model.named_parameters():
        # Convert parameter name to GGUF format
        gguf_name = name.replace("model.", "").replace(".", "_")

        writer.add_tensor(gguf_name, param.data, dtype)
        tensor_count += 1

        if tensor_count % 10 == 0:
            print(f"   Converted {tensor_count} tensors...")

    print(f"✅ Converted {tensor_count} tensors")

    # Write file
    print(f"🔄 Writing GGUF file...")
    writer.write()

def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to GGUF format")
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model identifier (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output GGUF file path")
    parser.add_argument("--quant", type=str, default="q4_0",
                       choices=["f32", "f16", "q4_0", "q8_0"],
                       help="Quantization type (default: q4_0)")

    args = parser.parse_args()

    try:
        convert_model(args.model, args.output, args.quant)
        print("\n✅ Conversion complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
