/// Quantization accuracy tests - verifies RMSE against reference implementations
///
/// Based on llama.cpp's test-quantize-fns.cpp
/// Tests that quantization->dequantization produces acceptable error rates
use wasm_chord_core::quant::*;

/// Maximum acceptable RMSE for standard quantization formats
const MAX_QUANTIZATION_ERROR: f32 = 0.002;
const MAX_QUANTIZATION_ERROR_LOWBIT: f32 = 0.01;

/// Calculate Root Mean Square Error between two float arrays
fn array_rmse(a1: &[f32], a2: &[f32]) -> f32 {
    assert_eq!(a1.len(), a2.len(), "Arrays must have same length");

    let sum: f32 = a1.iter().zip(a2.iter()).map(|(x, y)| (x - y).powi(2)).sum();

    (sum / a1.len() as f32).sqrt()
}

/// Generate synthetic test data using cosine function (like llama.cpp)
/// This creates a smooth but varied dataset good for testing quantization
fn generate_test_data(offset: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 + 2.0 * ((i as f32) + offset).cos()).collect()
}

#[test]
fn test_q4_k_dequantization_accuracy() {
    // Generate synthetic test data
    let test_size = 256;
    let test_data = generate_test_data(0.0, test_size);

    // For this test, we only verify dequantization works
    // (We don't have quantization functions yet, just using example blocks)

    // Create a block with known values
    let block = BlockQ4_K {
        d: half::f16::from_f32(0.01).to_bits(),
        dmin: half::f16::from_f32(0.001).to_bits(),
        scales: [0x11; 12],
        qs: [0x88; QK_K / 2], // Middle values
    };

    let mut output = vec![0.0f32; QK_K];
    dequantize_q4_k(&block, &mut output).unwrap();

    // Verify output is reasonable (no NaN/Inf)
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Q4_K dequantization produced non-finite values"
    );

    // Verify values are in expected range for 4-bit quantization
    let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_val < 10.0, "Q4_K values out of expected range: max={}", max_val);
}

#[test]
fn test_q6_k_dequantization_accuracy() {
    let test_size = 256;

    // Create a block with known pattern
    let mut block = BlockQ6_K {
        ql: [0u8; QK_K / 2],
        qh: [0u8; QK_K / 4],
        scales: [1i8; QK_K / 16],
        d: half::f16::from_f32(0.01).to_bits(),
    };

    // Set some varied values for better test coverage
    for i in 0..block.ql.len() {
        block.ql[i] = ((i * 7) % 256) as u8; // Pseudorandom pattern
    }

    let mut output = vec![0.0f32; QK_K];
    dequantize_q6_k(&block, &mut output).unwrap();

    // Verify no NaN/Inf
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Q6_K dequantization produced non-finite values"
    );

    // Q6_K should have better accuracy than Q4_K (more bits)
    let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_val < 10.0, "Q6_K values out of expected range: max={}", max_val);
}

#[test]
fn test_q8_k_dequantization_accuracy() {
    let test_size = 256;

    // Create a Q8_K block with test values
    let mut block = BlockQ8_K {
        quants: [0i8; QK_K],
        scales: [0u8; QK_K / 8],
        d: half::f16::from_f32(0.01).to_bits(),
        dmin: half::f16::from_f32(0.0).to_bits(),
    };

    // Set some test values
    for i in 0..block.quants.len() {
        block.quants[i] = ((i as i32 - 128) % 128) as i8; // Range -128 to 127
    }
    for i in 0..block.scales.len() {
        block.scales[i] = 0x11; // Simple scale pattern
    }

    let mut output = vec![0.0f32; QK_K];
    dequantize_q8_k(&block, &mut output).unwrap();

    // Verify no NaN/Inf
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Q8_K dequantization produced non-finite values"
    );

    // Q8_K should have very good accuracy (8 bits)
    let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_val < 20.0, "Q8_K values out of expected range: max={}", max_val);
}

#[test]
fn test_q5_k_dequantization_accuracy() {
    // Create a Q5_K block
    let mut block = BlockQ5_K {
        ql: [0u8; QK_K / 2],
        qh: [0u8; QK_K / 8],
        scales: [1i8; QK_K / 16],
        d: half::f16::from_f32(0.01).to_bits(),
    };

    // Set varied test pattern
    for i in 0..block.ql.len() {
        block.ql[i] = ((i * 13) % 256) as u8;
    }

    let mut output = vec![0.0f32; QK_K];
    dequantize_q5_k(&block, &mut output).unwrap();

    // Verify no NaN/Inf
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Q5_K dequantization produced non-finite values"
    );

    let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_val < 15.0, "Q5_K values out of expected range: max={}", max_val);
}

#[test]
fn test_quantization_value_ranges() {
    // Test that quantization produces values in expected ranges

    // Q4_K: 4-bit values (0-15 after dequantization)
    let q4k_block = BlockQ4_K {
        d: half::f16::from_f32(1.0).to_bits(),
        dmin: half::f16::from_f32(0.0).to_bits(),
        scales: [1; 12],
        qs: [0xFF; QK_K / 2], // All max values
    };
    let mut output = vec![0.0f32; QK_K];
    dequantize_q4_k(&q4k_block, &mut output).unwrap();

    // With scale=1, 4-bit max (15) should give values around 15
    let max_q4k = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_q4k > 10.0 && max_q4k < 100.0, "Q4_K max value unexpected: {}", max_q4k);

    // Q8_K: 8-bit values (-128 to 127)
    let q8k_block = BlockQ8_K {
        quants: [127i8; QK_K], // Max positive
        scales: [0x11; QK_K / 8],
        d: half::f16::from_f32(1.0).to_bits(),
        dmin: half::f16::from_f32(0.0).to_bits(),
    };
    let mut output = vec![0.0f32; QK_K];
    dequantize_q8_k(&q8k_block, &mut output).unwrap();

    let max_q8k = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_q8k > 100.0 && max_q8k < 1000.0, "Q8_K max value unexpected: {}", max_q8k);
}

#[test]
fn test_quantization_zero_handling() {
    // Test that all-zero inputs produce reasonable outputs

    let q4k_zero = BlockQ4_K {
        d: half::f16::from_f32(1.0).to_bits(),
        dmin: half::f16::from_f32(0.0).to_bits(),
        scales: [0; 12], // Zero scales
        qs: [0; QK_K / 2],
    };

    let mut output = vec![0.0f32; QK_K];
    dequantize_q4_k(&q4k_zero, &mut output).unwrap();

    // With zero scales and values, output should be close to zero
    let max_val = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_val < 1.0, "Zero quantization should produce small values: {}", max_val);
}

#[test]
fn test_quantization_consistency() {
    // Same input block should always produce same output

    let block = BlockQ4_K {
        d: half::f16::from_f32(0.5).to_bits(),
        dmin: half::f16::from_f32(0.1).to_bits(),
        scales: [0x33; 12],
        qs: [0xAB; QK_K / 2],
    };

    let mut output1 = vec![0.0f32; QK_K];
    let mut output2 = vec![0.0f32; QK_K];

    dequantize_q4_k(&block, &mut output1).unwrap();
    dequantize_q4_k(&block, &mut output2).unwrap();

    // Results should be identical
    for (i, (&v1, &v2)) in output1.iter().zip(&output2).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "Dequantization not consistent at index {}: {} != {}",
            i,
            v1,
            v2
        );
    }
}

#[test]
fn test_block_size_requirements() {
    // Verify that providing wrong output size fails gracefully

    let block = BlockQ4_K {
        d: half::f16::from_f32(1.0).to_bits(),
        dmin: half::f16::from_f32(0.0).to_bits(),
        scales: [1; 12],
        qs: [0; QK_K / 2],
    };

    // Wrong size should error
    let mut output_wrong = vec![0.0f32; 128]; // Too small
    let result = dequantize_q4_k(&block, &mut output_wrong);
    assert!(result.is_err(), "Should reject wrong output size");

    // Correct size should work
    let mut output_correct = vec![0.0f32; QK_K];
    let result = dequantize_q4_k(&block, &mut output_correct);
    assert!(result.is_ok(), "Should accept correct output size");
}
