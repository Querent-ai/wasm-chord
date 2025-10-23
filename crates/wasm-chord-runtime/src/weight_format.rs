//! Weight format abstraction for fused kernel dispatch
//!
//! This module enables storing weights in their native quantized format,
//! allowing efficient dispatch to fused dequantization + matmul kernels.

use wasm_chord_core::quant::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K};

/// Weight storage format for efficient kernel dispatch
#[derive(Debug, Clone)]
pub enum WeightFormat {
    /// Full precision (F32)
    F32(Vec<f32>),

    /// Q4_K quantized (~0.5 bytes/element)
    Q4K(Vec<BlockQ4_K>),

    /// Q5_K quantized (~0.625 bytes/element)
    Q5K(Vec<BlockQ5_K>),

    /// Q6_K quantized (~0.75 bytes/element)
    Q6K(Vec<BlockQ6_K>),

    /// Q8_K quantized (~1 byte/element)
    Q8K(Vec<BlockQ8_K>),
}

impl WeightFormat {
    /// Get number of elements (dequantized count)
    pub fn len(&self) -> usize {
        match self {
            WeightFormat::F32(v) => v.len(),
            WeightFormat::Q4K(v) => v.len() * 256,
            WeightFormat::Q5K(v) => v.len() * 256,
            WeightFormat::Q6K(v) => v.len() * 256,
            WeightFormat::Q8K(v) => v.len() * 256,
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        match self {
            WeightFormat::F32(v) => v.is_empty(),
            WeightFormat::Q4K(v) => v.is_empty(),
            WeightFormat::Q5K(v) => v.is_empty(),
            WeightFormat::Q6K(v) => v.is_empty(),
            WeightFormat::Q8K(v) => v.is_empty(),
        }
    }

    /// Create empty F32 weight with given size
    pub fn new_f32(size: usize) -> Self {
        WeightFormat::F32(vec![0.0; size])
    }

    /// Get format name
    pub fn format_name(&self) -> &'static str {
        match self {
            WeightFormat::F32(_) => "F32",
            WeightFormat::Q4K(_) => "Q4_K",
            WeightFormat::Q5K(_) => "Q5_K",
            WeightFormat::Q6K(_) => "Q6_K",
            WeightFormat::Q8K(_) => "Q8_K",
        }
    }
}

impl Default for WeightFormat {
    fn default() -> Self {
        WeightFormat::F32(Vec::new())
    }
}
