//! Extensions to TensorLoader for loading quantized weights
//!
//! Provides helpers to load weights in their native quantized format
//! for use with fused kernels.

use std::io::{Read, Seek};
use wasm_chord_core::error::{Error, Result};
use wasm_chord_core::formats::gguf::GGUFParser;
use wasm_chord_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K};
use wasm_chord_core::tensor::DataType;
use wasm_chord_core::TensorMetadata;

use crate::weight_format::WeightFormat;

/// Load a tensor in its optimal format (quantized blocks or F32)
///
/// This function loads Q4_K/Q5_K/Q6_K/Q8_K weights as quantized blocks,
/// enabling efficient fused kernel dispatch. Other formats are dequantized to F32.
pub fn load_weight_optimal<R: Read + Seek>(
    tensor_name: &str,
    metadata: &TensorMetadata,
    parser: &mut GGUFParser<R>,
    _data_offset: u64,
) -> Result<WeightFormat> {
    // Calculate absolute offset
    // metadata.offset is absolute from file start, not relative to tensor data section
    let absolute_offset = metadata.offset;

    // Read raw tensor data
    let raw_data = parser.read_tensor_data(absolute_offset, metadata.size_bytes)?;

    // Return appropriate format based on dtype
    match metadata.desc.dtype {
        DataType::F32 => {
            // F32: Parse as float array
            let mut result = Vec::with_capacity(metadata.desc.element_count());
            for chunk in raw_data.chunks_exact(4) {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                result.push(f32::from_le_bytes(bytes));
            }
            Ok(WeightFormat::F32(result))
        }
        DataType::Q4_K => {
            // Q4_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q4K(blocks))
        }
        DataType::Q5_K => {
            // Q5_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ5_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q5K(blocks))
        }
        DataType::Q6_K => {
            // Q6_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ6_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q6K(blocks))
        }
        DataType::Q8_K => {
            // Q8_K: Keep as quantized blocks
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_K>();
            let num_blocks = raw_data.len() / BLOCK_SIZE;
            let mut blocks = Vec::with_capacity(num_blocks);

            for block_data in raw_data.chunks_exact(BLOCK_SIZE) {
                let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ8_K) };
                blocks.push(block);
            }
            Ok(WeightFormat::Q8K(blocks))
        }
        _ => {
            // For unsupported formats, return error
            // (TensorLoader will handle dequantization for Q4_0, Q8_0, etc.)
            Err(Error::UnsupportedDataType(format!(
                "Tensor {} has unsupported dtype {:?} for optimal loading. Use TensorLoader::load_tensor() instead.",
                tensor_name, metadata.desc.dtype
            )))
        }
    }
}
