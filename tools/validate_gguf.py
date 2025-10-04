#!/usr/bin/env python3
"""
Validate GGUF files for wasm-chord

Checks:
- Magic number and version
- Metadata integrity
- Tensor info correctness
- File size consistency

Usage:
    python validate_gguf.py model.gguf
"""

import argparse
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

def read_string(f):
    """Read a length-prefixed UTF-8 string"""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def validate_gguf(file_path: str) -> bool:
    """Validate a GGUF file"""

    print(f"🔍 Validating: {file_path}")

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False

    file_size = Path(file_path).stat().st_size
    print(f"   File size: {file_size / (1024**2):.2f} MB")

    try:
        with open(file_path, 'rb') as f:
            # Read and validate header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                print(f"❌ Invalid magic number: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})")
                return False
            print(f"✅ Magic number: 0x{magic:08X}")

            version = struct.unpack('<I', f.read(4))[0]
            if version != GGUF_VERSION:
                print(f"⚠️  Version: {version} (expected {GGUF_VERSION})")
            else:
                print(f"✅ Version: {version}")

            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            print(f"✅ Tensor count: {tensor_count}")
            print(f"✅ Metadata count: {metadata_kv_count}")

            # Read metadata
            print(f"\n📋 Metadata:")
            metadata = {}
            for i in range(metadata_kv_count):
                key = read_string(f)

                # Read value type
                value_type = struct.unpack('<I', f.read(4))[0]

                if value_type == 4:  # String
                    value = read_string(f)
                elif value_type == 6:  # Uint32
                    value = struct.unpack('<I', f.read(4))[0]
                elif value_type == 2:  # Float32
                    value = struct.unpack('<f', f.read(4))[0]
                else:
                    print(f"⚠️  Unknown value type: {value_type} for key: {key}")
                    continue

                metadata[key] = value

                # Print important metadata
                if not key.startswith("tokenizer.ggml.token."):
                    print(f"   {key}: {value}")

            # Read tensor info
            print(f"\n🔢 Tensors:")
            tensors = []
            for i in range(tensor_count):
                name = read_string(f)
                ndim = struct.unpack('<I', f.read(4))[0]

                dims = []
                for _ in range(ndim):
                    dim = struct.unpack('<Q', f.read(8))[0]
                    dims.append(dim)

                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]

                tensors.append({
                    'name': name,
                    'dims': dims,
                    'dtype': dtype,
                    'offset': offset
                })

                if i < 10:  # Print first 10 tensors
                    dtype_name = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0"}.get(dtype, f"UNKNOWN({dtype})")
                    print(f"   [{i}] {name}: {dims} ({dtype_name})")

            if tensor_count > 10:
                print(f"   ... and {tensor_count - 10} more tensors")

            # Validate file size
            data_start = f.tell()
            # Align to 32 bytes
            padding = (32 - (data_start % 32)) % 32
            data_start += padding

            print(f"\n📊 File structure:")
            print(f"   Header end: {data_start} bytes")
            print(f"   Data size: {file_size - data_start} bytes")

            if file_size < data_start:
                print(f"❌ File too small! Missing tensor data")
                return False

            print(f"\n✅ Validation passed!")
            return True

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate GGUF files")
    parser.add_argument("file", type=str, help="GGUF file to validate")

    args = parser.parse_args()

    success = validate_gguf(args.file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
