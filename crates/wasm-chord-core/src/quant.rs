//! Quantization and dequantization primitives

use crate::error::{Error, Result};
use crate::tensor::DataType;

/// Block size for group-wise quantization
pub const Q4_BLOCK_SIZE: usize = 32;
pub const Q8_BLOCK_SIZE: usize = 32;
pub const QK_K: usize = 256; // K-quants use 256-element super-blocks

/// Q4_0 block: 32 4-bit values + 1 f16 scale
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub scale: f32,
    pub quants: [u8; Q4_BLOCK_SIZE / 2], // 16 bytes (2 values per byte)
}

/// Q8_0 block: 32 8-bit values + 1 f16 scale
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub scale: f32,
    pub quants: [i8; Q8_BLOCK_SIZE],
}

/// Q6_K block: 256 6-bit values in a super-block
/// Structure based on ggml implementation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6_K {
    pub ql: [u8; QK_K / 2],      // Lower 4 bits of 6-bit quants
    pub qh: [u8; QK_K / 4],      // Upper 2 bits of 6-bit quants
    pub scales: [i8; QK_K / 16], // 16 scales per block
    pub d: f32,                  // Delta (super-scale)
}

/// Dequantize Q4_0 block to f32
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32]) -> Result<()> {
    if output.len() != Q4_BLOCK_SIZE {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            Q4_BLOCK_SIZE
        )));
    }

    let scale = block.scale;

    for i in 0..Q4_BLOCK_SIZE / 2 {
        let byte = block.quants[i];

        // Lower 4 bits
        let v0 = ((byte & 0x0F) as i8) - 8;
        output[i * 2] = v0 as f32 * scale;

        // Upper 4 bits
        let v1 = ((byte >> 4) as i8) - 8;
        output[i * 2 + 1] = v1 as f32 * scale;
    }

    Ok(())
}

/// Dequantize Q8_0 block to f32
pub fn dequantize_q8_0(block: &BlockQ8_0, output: &mut [f32]) -> Result<()> {
    if output.len() != Q8_BLOCK_SIZE {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            Q8_BLOCK_SIZE
        )));
    }

    let scale = block.scale;

    for (i, &quant) in block.quants.iter().enumerate().take(Q8_BLOCK_SIZE) {
        output[i] = quant as f32 * scale;
    }

    Ok(())
}

/// Dequantize Q6_K block to f32
/// Q6_K stores 256 values as 6-bit quantized values
#[allow(clippy::needless_range_loop)]
pub fn dequantize_q6_k(block: &BlockQ6_K, output: &mut [f32]) -> Result<()> {
    if output.len() != QK_K {
        return Err(Error::InvalidShape(format!("Output buffer must be {} elements", QK_K)));
    }

    let d = block.d;

    // Reconstruct 6-bit values from lower 4 bits (ql) and upper 2 bits (qh)
    for i in 0..QK_K {
        // Get scale for this group of 16 elements
        let scale_idx = i / 16;
        let scale = block.scales[scale_idx] as f32;

        // Get lower 4 bits
        let ql_byte_idx = i / 2;
        let ql_shift = (i % 2) * 4;
        let ql = (block.ql[ql_byte_idx] >> ql_shift) & 0x0F;

        // Get upper 2 bits
        let qh_byte_idx = i / 4;
        let qh_shift = (i % 4) * 2;
        let qh = (block.qh[qh_byte_idx] >> qh_shift) & 0x03;

        // Combine to get 6-bit value (0-63)
        let q6 = (qh << 4) | ql;

        // Dequantize: apply scale offset, then multiply by scales
        let q = (q6 as i8) - 32; // Center around 0 (range: -32 to 31)
        output[i] = d * scale * (q as f32);
    }

    Ok(())
}

/// Get block size for quantization type
pub fn get_block_size(dtype: DataType) -> Option<usize> {
    match dtype {
        DataType::Q4_0 | DataType::Q4_1 => Some(Q4_BLOCK_SIZE),
        DataType::Q8_0 | DataType::Q8_1 => Some(Q8_BLOCK_SIZE),
        DataType::Q6_K => Some(QK_K),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_dequant() {
        let block = BlockQ4_0 {
            scale: 0.5,
            quants: [0x10; 16], // All values = 1, 0
        };

        let mut output = [0.0f32; Q4_BLOCK_SIZE];
        dequantize_q4_0(&block, &mut output).unwrap();

        // 0x10 -> lower = 0, upper = 1
        // After offset -8: lower = -8, upper = -7
        assert_eq!(output[0], -8.0 * 0.5);
        assert_eq!(output[1], -7.0 * 0.5);
    }

    #[test]
    fn test_q8_dequant() {
        let block = BlockQ8_0 { scale: 0.25, quants: [10i8; Q8_BLOCK_SIZE] };

        let mut output = [0.0f32; Q8_BLOCK_SIZE];
        dequantize_q8_0(&block, &mut output).unwrap();

        assert_eq!(output[0], 10.0 * 0.25);
    }
}
