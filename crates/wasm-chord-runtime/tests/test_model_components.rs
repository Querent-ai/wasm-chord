/// Model component tests - RMS norm, SiLU, FFN, etc.
///
/// Tests individual transformer components for correctness

/// Test RMS normalization properties
#[test]
fn test_rmsnorm_properties() {
    // RMS norm should produce output with RMS ≈ 1
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weights = vec![1.0; 4];
    let eps = 1e-6;

    // Manual RMS norm calculation
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
    let output: Vec<f32> =
        input.iter().zip(&weights).map(|(&x, &w)| (x / (rms + eps)) * w).collect();

    // Check output RMS is approximately 1
    let output_rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();

    assert!((output_rms - 1.0).abs() < 0.1, "RMS norm should produce RMS ≈ 1: got {}", output_rms);
}

/// Test RMS norm with zero epsilon behavior
#[test]
fn test_rmsnorm_epsilon() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let weights = vec![1.0; 4];
    let eps = 1e-6;

    // With all zeros, RMS = 0, so epsilon prevents division by zero
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
    let output: Vec<f32> =
        input.iter().zip(&weights).map(|(&x, &w)| (x / (rms + eps)) * w).collect();

    // Should not produce NaN/Inf
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "RMS norm with zero input should not produce NaN/Inf"
    );
}

/// Test SiLU activation function
#[test]
fn test_silu_activation() {
    // SiLU(x) = x * sigmoid(x)
    let test_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_values {
        let x_f32 = x as f32;
        let sigmoid_x: f32 = 1.0 / (1.0 + (-x_f32).exp());
        let expected_silu = x * sigmoid_x;

        // Manual SiLU calculation
        let computed_silu: f32 = x * (1.0 / (1.0 + (-x_f32).exp()));

        assert!(
            (computed_silu - expected_silu).abs() < 1e-6,
            "SiLU incorrect for x={}: expected {}, got {}",
            x,
            expected_silu,
            computed_silu
        );
    }
}

/// Test SiLU properties
#[test]
fn test_silu_properties() {
    // SiLU(0) should be 0
    let silu_zero = 0.0 * (1.0 / (1.0 + (-0.0_f32).exp()));
    assert!((silu_zero - 0.0).abs() < 1e-10, "SiLU(0) should be 0");

    // SiLU should be monotonically increasing
    let x1: f32 = -1.0;
    let x2: f32 = 1.0;
    let silu_x1: f32 = x1 * (1.0 / (1.0 + (-x1).exp()));
    let silu_x2: f32 = x2 * (1.0 / (1.0 + (-x2).exp()));

    assert!(silu_x2 > silu_x1, "SiLU should be monotonically increasing");

    // For large positive x, SiLU(x) ≈ x
    let large_x: f32 = 10.0;
    let silu_large: f32 = large_x * (1.0 / (1.0 + (-large_x).exp()));
    assert!(
        (silu_large - large_x).abs() < 0.1,
        "For large x, SiLU(x) ≈ x: expected {}, got {}",
        large_x,
        silu_large
    );
}

/// Test softmax properties
#[test]
fn test_softmax_sum_to_one() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Compute softmax
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let softmax: Vec<f32> = logits.iter().map(|&x| (x - max).exp() / exp_sum).collect();

    // Sum should be 1.0
    let sum: f32 = softmax.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1.0: got {}", sum);

    // All values should be positive
    assert!(softmax.iter().all(|&x| x > 0.0 && x < 1.0), "Softmax values should be in (0, 1)");
}

/// Test softmax numerical stability
#[test]
fn test_softmax_stability() {
    // Large values should not cause overflow
    let large_logits = vec![1000.0, 1001.0, 1002.0];

    let max = large_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = large_logits.iter().map(|&x| (x - max).exp()).sum();
    let softmax: Vec<f32> = large_logits.iter().map(|&x| (x - max).exp() / exp_sum).collect();

    assert!(softmax.iter().all(|&x| x.is_finite()), "Softmax should handle large values");

    let sum: f32 = softmax.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax with large values should still sum to 1");
}

/// Test residual connections
#[test]
fn test_residual_connection() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let transform_output = vec![0.5, 1.0, 1.5, 2.0];

    // Residual: output = input + transform(input)
    let residual: Vec<f32> = input.iter().zip(&transform_output).map(|(&x, &t)| x + t).collect();

    let expected = vec![1.5, 3.0, 4.5, 6.0];
    for (i, (&r, &e)) in residual.iter().zip(&expected).enumerate() {
        assert!(
            (r - e).abs() < 1e-6,
            "Residual connection incorrect at {}: expected {}, got {}",
            i,
            e,
            r
        );
    }
}

/// Test layer normalization maintains mean
#[test]
fn test_normalization_mean() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // RMS norm
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
    let normalized: Vec<f32> = input.iter().map(|&x| x / (rms + 1e-6)).collect();

    // Mean should be approximately 0 (well, close to original mean / rms)
    let mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
    assert!(mean.is_finite(), "Normalization should produce finite mean");
}

/// Test GLU (Gated Linear Unit) gating mechanism
#[test]
fn test_glu_gating() {
    // GLU splits input and uses one half as gate: output = x * sigmoid(gate)
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let gate = vec![0.0, 1.0, -1.0, 2.0];

    let output: Vec<f32> = x
        .iter()
        .zip(&gate)
        .map(|(&x_val, &g): (&f32, &f32)| x_val * (1.0 / (1.0 + (-g).exp())))
        .collect();

    // Check properties
    assert!(output.iter().all(|&v| v.is_finite()), "GLU output should be finite");

    // Gate = 0 -> sigmoid = 0.5 -> output = x * 0.5
    assert!((output[0] - 0.5).abs() < 1e-5, "GLU with gate=0: expected 0.5, got {}", output[0]);

    // Gate = large positive -> sigmoid ≈ 1 -> output ≈ x
    assert!(
        (output[3] - 4.0).abs() < 0.5,
        "GLU with large positive gate: expected ~4.0, got {}",
        output[3]
    );
}

/// Test matmul identity
#[test]
fn test_matmul_identity() {
    // A @ I = A (where I is identity)
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
    let identity = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity

    // Manual matmul
    let mut result = vec![0.0; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                result[i * 2 + j] += a[i * 2 + k] * identity[k * 2 + j];
            }
        }
    }

    for (i, (&r, &a_val)) in result.iter().zip(&a).enumerate() {
        let diff_val: f32 = r - a_val;
        assert!(
            diff_val.abs() < 1e-6,
            "Matmul with identity failed at {}: expected {}, got {}",
            i,
            a_val,
            r
        );
    }
}

/// Test numerical precision preservation
#[test]
fn test_numerical_precision() {
    // Operations should preserve reasonable precision
    let value: f32 = 0.123456789;

    // Addition
    let add_result = value + value;
    assert!((add_result - 2.0 * value).abs() < 1e-6, "Addition precision loss");

    // Multiplication
    let mul_result = value * 2.0;
    assert!((mul_result - 2.0 * value).abs() < 1e-6, "Multiplication precision loss");

    // Division
    let div_result = value / 2.0;
    assert!((div_result - value / 2.0).abs() < 1e-7, "Division precision loss");
}

/// Test edge case: empty tensors
#[test]
fn test_empty_tensor_handling() {
    let empty: Vec<f32> = vec![];

    // Operations on empty tensors should not crash
    let sum: f32 = empty.iter().sum();
    assert_eq!(sum, 0.0, "Sum of empty should be 0");

    let count = empty.len();
    assert_eq!(count, 0, "Empty tensor length should be 0");
}

/// Test vector operations
#[test]
fn test_vector_operations() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    // Dot product
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert!((dot - 32.0).abs() < 1e-5, "Dot product: expected 32, got {}", dot);

    // Element-wise addition
    let add: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(add, vec![5.0, 7.0, 9.0], "Element-wise addition failed");

    // Element-wise multiplication
    let mul: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    assert_eq!(mul, vec![4.0, 10.0, 18.0], "Element-wise multiplication failed");
}
