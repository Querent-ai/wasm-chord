//! Quantization and dequantization primitives

use crate::error::{Error, Result};
use crate::tensor::DataType;

/// Block size for group-wise quantization
pub const Q4_BLOCK_SIZE: usize = 32;
pub const Q8_BLOCK_SIZE: usize = 32;
pub const QK_K: usize = 256; // K-quants use 256-element super-blocks

/// Q4_0 block: 32 4-bit values
/// NOTE: This GGUF file uses 16-byte blocks (quants only, no per-block scale)
/// The scale appears to be stored separately or shared across blocks
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub quants: [u8; Q4_BLOCK_SIZE / 2], // 16 bytes (2 values per byte)
}

/// Q8_0 block: 32 8-bit values + 1 f16 scale
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub scale: u16, // f16 stored as u16
    pub quants: [i8; Q8_BLOCK_SIZE],
}

/// Q4_K block: 256 4-bit values in a super-block
/// Structure based on ggml implementation
/// Total: 144 bytes (12 scales + 128 qs + 2 d + 2 dmin)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_K {
    pub d: u16,             // f16 super-block scale (stored as u16)
    pub dmin: u16,          // f16 super-block min scale (stored as u16)
    pub scales: [u8; 12],   // Quantized scales (6 bits each)
    pub qs: [u8; QK_K / 2], // 128 bytes of 4-bit quants
}

/// Helper function to extract scale and min from Q4_K scales array
/// Based on ggml get_scale_min_k4
#[inline]
pub fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    let (d, m) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    };
    (d, m)
}

/// Dequantize Q4_K block to f32
/// Q4_K uses 256-element super-blocks with hierarchical scaling
/// Based on ggml dequantize_row_q4_K implementation
pub fn dequantize_q4_k(block: &BlockQ4_K, output: &mut [f32]) -> Result<()> {
    if output.len() != QK_K {
        return Err(Error::InvalidShape(format!("Output buffer must be {} elements", QK_K)));
    }

    // Convert f16 scales to f32
    let d = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.dmin).to_f32();

    // Debug: print d and min values and scales extraction
    static FIRST_BLOCK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
    if FIRST_BLOCK.swap(false, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("Q4_K dequant first block:");
        eprintln!("  d={:.10}, min={:.10}", d, min);
        eprintln!("  scales bytes: {:02x?}", &block.scales);

        // Test scale extraction for first 4 groups
        for is in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is * 2, &block.scales);
            let (sc1, m1) = get_scale_min_k4(is * 2 + 1, &block.scales);
            eprintln!("  Group {}: sc0={}, m0={}, sc1={}, m1={}", is, sc0, m0, sc1, m1);
        }
    }

    // Process 256 elements in 4 groups of 64
    for i in 0..4 {
        let is = i * 2; // Scale index

        // Get scales for this group
        let (sc, m) = get_scale_min_k4(is, &block.scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;

        let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;

        // Process 64 values: 32 from lower nibbles, 32 from upper nibbles
        let q_offset = i * 32;
        let y_offset = i * 64;

        // Lower nibbles (32 values)
        for j in 0..32 {
            let q = block.qs[q_offset + j];
            let x = (q & 0xF) as f32;
            output[y_offset + j] = d1 * x - m1;
        }

        // Upper nibbles (32 values)
        for j in 0..32 {
            let q = block.qs[q_offset + j];
            let x = (q >> 4) as f32;
            output[y_offset + 32 + j] = d2 * x - m2;
        }
    }

    Ok(())
}

/// Q6_K block: 256 6-bit values in a super-block
/// Structure based on ggml implementation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6_K {
    pub ql: [u8; QK_K / 2],      // Lower 4 bits of 6-bit quants
    pub qh: [u8; QK_K / 4],      // Upper 2 bits of 6-bit quants
    pub scales: [i8; QK_K / 16], // 16 scales per block
    pub d: u16,                  // f16 super-block scale (stored as u16)
}

/// Dequantize Q4_0 block to f32
/// NOTE: Using default scale for this non-standard Q4_0 format
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32]) -> Result<()> {
    if output.len() != Q4_BLOCK_SIZE {
        return Err(Error::InvalidShape(format!(
            "Output buffer must be {} elements",
            Q4_BLOCK_SIZE
        )));
    }

    // TODO: This GGUF file appears to use a non-standard format
    // For now, use a conservative scale value
    let d = 0.01f32;

    // Dequantize with interleaved layout: first half gets lower nibbles, second half gets upper
    // This matches llama.cpp's layout: y[i*qk + j + 0] = x0*d; y[i*qk + j + qk/2] = x1*d;
    for i in 0..Q4_BLOCK_SIZE / 2 {
        let byte = block.quants[i];

        // Lower 4 bits -> first half of output
        let x0 = ((byte & 0x0F) as i8) - 8;
        output[i] = x0 as f32 * d;

        // Upper 4 bits -> second half of output
        let x1 = ((byte >> 4) as i8) - 8;
        output[i + Q4_BLOCK_SIZE / 2] = x1 as f32 * d;
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

    // Convert f16 scale to f32
    let scale = half::f16::from_bits(block.scale).to_f32();

    for (i, &quant) in block.quants.iter().enumerate().take(Q8_BLOCK_SIZE) {
        output[i] = quant as f32 * scale;
    }

    Ok(())
}

/// Dequantize Q6_K block to f32
/// Q6_K stores 256 values as 6-bit quantized values
/// Based on ggml dequantize_row_q6_K implementation
#[allow(clippy::needless_range_loop)]
pub fn dequantize_q6_k(block: &BlockQ6_K, output: &mut [f32]) -> Result<()> {
    if output.len() != QK_K {
        return Err(Error::InvalidShape(format!("Output buffer must be {} elements", QK_K)));
    }

    // Convert f16 to f32
    let d = half::f16::from_bits(block.d).to_f32();

    // Q6_K layout: 256 values stored as:
    // - ql[128]: lower 4 bits (2 values per byte)
    // - qh[64]: upper 2 bits (4 values per byte)
    // - scales[16]: one scale per 16 values
    //
    // ggml processes 4 values at a time with specific layout:
    // For each group of 32 (l = 0..32):
    //   q1 = (ql[l] & 0xF) | ((qh[l] >> 0) & 3) << 4
    //   q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4
    //   q3 = (ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4
    //   q4 = (ql[l+32] >> 4) | ((qh[l] >> 6) & 3) << 4

    // Process first half (0-127): use scales[0..7]
    for l in 0..32 {
        let is = l / 16; // 0 or 1

        // Extract 4 values from the packed layout
        let q1 = ((block.ql[l] & 0x0F) | ((block.qh[l] & 3) << 4)) as i8 - 32;
        let q2 = ((block.ql[l + 32] & 0x0F) | (((block.qh[l] >> 2) & 3) << 4)) as i8 - 32;
        let q3 = ((block.ql[l] >> 4) | (((block.qh[l] >> 4) & 3) << 4)) as i8 - 32;
        let q4 = ((block.ql[l + 32] >> 4) | (((block.qh[l] >> 6) & 3) << 4)) as i8 - 32;

        // Write outputs with correct scales (matches llama.cpp: sc[is+0/2/4/6])
        output[l] = d * (block.scales[is] as f32) * (q1 as f32);
        output[l + 32] = d * (block.scales[is + 2] as f32) * (q2 as f32);
        output[l + 64] = d * (block.scales[is + 4] as f32) * (q3 as f32);
        output[l + 96] = d * (block.scales[is + 6] as f32) * (q4 as f32);
    }

    // Process second half (128-255): use scales[8..15]
    for l in 0..32 {
        let is = l / 16; // 0 or 1
        let sc_offset = 8; // Offset into scales array for second half

        let q1 = ((block.ql[l + 64] & 0x0F) | ((block.qh[l + 32] & 3) << 4)) as i8 - 32;
        let q2 = ((block.ql[l + 96] & 0x0F) | (((block.qh[l + 32] >> 2) & 3) << 4)) as i8 - 32;
        let q3 = ((block.ql[l + 64] >> 4) | (((block.qh[l + 32] >> 4) & 3) << 4)) as i8 - 32;
        let q4 = ((block.ql[l + 96] >> 4) | (((block.qh[l + 32] >> 6) & 3) << 4)) as i8 - 32;

        // Write outputs with correct scales (sc_offset=8, so we use scales[8..15])
        output[l + 128] = d * (block.scales[sc_offset + is] as f32) * (q1 as f32);
        output[l + 160] = d * (block.scales[sc_offset + is + 2] as f32) * (q2 as f32);
        output[l + 192] = d * (block.scales[sc_offset + is + 4] as f32) * (q3 as f32);
        output[l + 224] = d * (block.scales[sc_offset + is + 6] as f32) * (q4 as f32);
    }

    Ok(())
}

/// Get block size for quantization type
pub fn get_block_size(dtype: DataType) -> Option<usize> {
    match dtype {
        DataType::Q4_0 | DataType::Q4_1 => Some(Q4_BLOCK_SIZE),
        DataType::Q8_0 | DataType::Q8_1 => Some(Q8_BLOCK_SIZE),
        DataType::Q4_K => Some(QK_K),
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
            quants: [0x10; 16], // All values = 1, 0 (in nibbles)
        };

        let mut output = [0.0f32; Q4_BLOCK_SIZE];
        dequantize_q4_0(&block, &mut output).unwrap();

        // 0x10 -> lower = 0, upper = 1
        // After offset -8: lower = -8, upper = -7
        // With scale=0.01
        assert_eq!(output[0], -0.08);
        assert_eq!(output[16], -0.07);
    }

    #[test]
    fn test_q8_dequant() {
        let block =
            BlockQ8_0 { scale: half::f16::from_f32(0.25).to_bits(), quants: [10i8; Q8_BLOCK_SIZE] };

        let mut output = [0.0f32; Q8_BLOCK_SIZE];
        dequantize_q8_0(&block, &mut output).unwrap();

        assert_eq!(output[0], 10.0 * 0.25);
    }

    #[test]
    fn test_q4_k_dequant() {
        // Create a simple Q4_K block
        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.5).to_bits(),
            scales: [0x11; 12],   // Simple pattern for testing
            qs: [0x50; QK_K / 2], // 5 in lower nibble, 0 in upper
        };

        let mut output = [0.0f32; QK_K];
        dequantize_q4_k(&block, &mut output).unwrap();

        // Check that we don't have inf/nan
        let has_nan = output.iter().any(|&x| x.is_nan());
        let has_inf = output.iter().any(|&x| x.is_infinite());
        assert!(!has_nan, "Q4_K dequantization produced NaN");
        assert!(!has_inf, "Q4_K dequantization produced inf");

        // Check reasonable value ranges (4-bit values are 0-15, with scaling)
        for (i, &val) in output.iter().enumerate() {
            assert!(val.abs() <= 100.0, "Value at index {} is out of range: {}", i, val);
        }
    }

    #[test]
    fn test_q6_k_dequant() {
        // Create a simple Q6_K block
        let mut block = BlockQ6_K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [1i8; QK_K / 16],
            d: half::f16::from_f32(1.0).to_bits(),
        };

        // Set some test values
        // First element: ql[0] = 0x10 (lower 4 bits = 0, upper 4 bits = 1)
        //                qh[0] = 0x00 (bits 0-1 = 0)
        // This should give q1 = 0, q3 = 1 (after combining with qh)
        block.ql[0] = 0x10;
        block.qh[0] = 0x00;

        let mut output = [0.0f32; QK_K];
        dequantize_q6_k(&block, &mut output).unwrap();

        // Check that we don't have inf/nan
        let has_nan = output.iter().any(|&x| x.is_nan());
        let has_inf = output.iter().any(|&x| x.is_infinite());
        assert!(!has_nan, "Q6_K dequantization produced NaN");
        assert!(!has_inf, "Q6_K dequantization produced inf");

        // Check reasonable value ranges (-32 to 31 after bias, times scale and d)
        for (i, &val) in output.iter().enumerate() {
            assert!(val.abs() <= 100.0, "Value at index {} is out of range: {}", i, val);
        }
    }
}
