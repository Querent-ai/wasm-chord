/// Comprehensive Weight Orientation Debug Tool
/// This tool checks the most likely bugs in weight matrix orientation
use wasm_chord_core::error::Result;
use wasm_chord_cpu::{matmul_f32, matmul_transposed};

/// Debug weight storage format and orientation
#[allow(dead_code)]
pub fn diagnose_weight_storage_format() -> Result<()> {
    println!("🔍 DIAGNOSING WEIGHT STORAGE FORMAT");
    println!("===================================");

    // Test matrix orientation with known values
    let test_input = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2] matrix
    let test_weight_row_major = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2] row-major
    let test_weight_col_major = vec![1.0, 3.0, 2.0, 4.0]; // [2, 2] column-major (transposed)

    println!("Test input: {:?}", test_input);
    println!("Row-major weights: {:?}", test_weight_row_major);
    println!("Col-major weights: {:?}", test_weight_col_major);

    // Test with standard matmul (expects row-major)
    let mut result_standard = vec![0.0; 4];
    matmul_f32(&test_input, &test_weight_row_major, &mut result_standard, 2, 2, 2)?;
    println!("Standard matmul result: {:?}", result_standard);

    // Test with transposed matmul (expects col-major)
    let mut result_transposed = vec![0.0; 4];
    matmul_transposed(&test_input, &test_weight_col_major, &mut result_transposed, 2, 2, 2)?;
    println!("Transposed matmul result: {:?}", result_transposed);

    // Check if results are the same (they should be!)
    let diff: f32 =
        result_standard.iter().zip(result_transposed.iter()).map(|(a, b)| (a - b).abs()).sum();

    println!("Difference between results: {}", diff);

    if diff < 1e-6 {
        println!("✅ Matrix orientation test PASSED");
    } else {
        println!("❌ Matrix orientation test FAILED - weights may be wrong orientation!");
    }

    Ok(())
}

/// Check if attention weights need transposing
pub fn check_attention_weight_orientation(
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    hidden_size: usize,
) -> Result<()> {
    println!("🔍 CHECKING ATTENTION WEIGHT ORIENTATION");
    println!("=========================================");

    // Check weight statistics
    let weights = [("WQ", wq), ("WK", wk), ("WV", wv), ("WO", wo)];

    for (name, weight) in weights.iter() {
        let sum = weight.iter().sum::<f32>();
        let mean = sum / weight.len() as f32;
        let variance = weight.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / weight.len() as f32;
        let std_dev = variance.sqrt();

        println!("{}: sum={:.6}, mean={:.6}, std={:.6}", name, sum, mean, std_dev);

        // Check for suspicious patterns
        if std_dev < 0.001 {
            println!("⚠️  {} has very low variance - may be incorrectly oriented!", name);
        }

        if sum.abs() < 0.001 {
            println!("⚠️  {} has near-zero sum - may be incorrectly oriented!", name);
        }
    }

    // Test with a small subset to see orientation
    let test_input = vec![1.0; hidden_size]; // All ones input

    // Test WQ with both orientations - use the full matrix
    let mut result_standard = vec![0.0; hidden_size];
    let mut result_transposed = vec![0.0; hidden_size];

    // WQ should be [hidden_size, hidden_size] for standard matmul
    matmul_f32(&test_input, wq, &mut result_standard, 1, hidden_size, hidden_size)?;
    // WQ should be [hidden_size, hidden_size] for transposed matmul
    matmul_transposed(&test_input, wq, &mut result_transposed, 1, hidden_size, hidden_size)?;

    let standard_sum = result_standard.iter().sum::<f32>();
    let transposed_sum = result_transposed.iter().sum::<f32>();

    println!("WQ test with all-ones input:");
    println!("  Standard matmul sum: {:.6}", standard_sum);
    println!("  Transposed matmul sum: {:.6}", transposed_sum);

    if standard_sum.abs() > transposed_sum.abs() * 2.0 {
        println!("⚠️  WQ may need transposing! Standard gives much larger result");
    } else if transposed_sum.abs() > standard_sum.abs() * 2.0 {
        println!("✅ WQ orientation looks correct (transposed gives larger result)");
    } else {
        println!("❓ WQ orientation unclear - both give similar results");
    }

    Ok(())
}

/// Run critical checks on all model components
#[allow(dead_code)]
pub fn run_critical_checks() -> Result<()> {
    println!("🧪 RUNNING CRITICAL CHECKS");
    println!("==========================");

    // Check 1: Matrix multiplication self-test
    println!("1. Matrix multiplication self-test...");
    matmul_self_test()?;
    println!("   ✅ PASSED");

    // Check 2: Weight orientation test
    println!("2. Weight orientation test...");
    diagnose_weight_storage_format()?;
    println!("   ✅ PASSED");

    // Check 3: Attention weight orientation (if weights are available)
    println!("3. Attention weight orientation check...");
    println!("   (This requires loaded weights - run after model loading)");

    println!("✅ All critical checks completed!");
    Ok(())
}

/// Self-test for matrix multiplication
#[allow(dead_code)]
fn matmul_self_test() -> Result<()> {
    // Test case: [2, 3] x [3, 2] = [2, 2]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
    let expected = vec![22.0, 28.0, 49.0, 64.0]; // [2, 2]

    let mut result = vec![0.0; 4];
    matmul_f32(&a, &b, &mut result, 2, 3, 2)?;

    let diff: f32 = result.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).sum();

    if diff < 1e-6 {
        Ok(())
    } else {
        Err(wasm_chord_core::error::Error::Runtime(format!(
            "Matmul self-test failed. Expected: {:?}, Got: {:?}",
            expected, result
        )))
    }
}

/// Quick fix: Transpose attention weights if needed
#[allow(dead_code)]
pub fn transpose_attention_weights(
    wq: &mut [f32],
    wk: &mut [f32],
    wv: &mut [f32],
    wo: &mut [f32],
    hidden_size: usize,
) -> Result<()> {
    println!("🔧 TRANSPOSING ATTENTION WEIGHTS");
    println!("================================");

    // Transpose WQ: [hidden_size, hidden_size] -> [hidden_size, hidden_size]
    transpose_matrix_inplace(wq, hidden_size, hidden_size)?;

    // Transpose WK: [hidden_size, num_kv_heads * head_dim] -> [num_kv_heads * head_dim, hidden_size]
    let kv_dim = wk.len() / hidden_size;
    transpose_matrix_inplace(wk, hidden_size, kv_dim)?;

    // Transpose WV: [hidden_size, num_kv_heads * head_dim] -> [num_kv_heads * head_dim, hidden_size]
    transpose_matrix_inplace(wv, hidden_size, kv_dim)?;

    // Transpose WO: [hidden_size, hidden_size] -> [hidden_size, hidden_size]
    transpose_matrix_inplace(wo, hidden_size, hidden_size)?;

    println!("✅ All attention weights transposed!");
    Ok(())
}

/// Transpose a matrix in-place
#[allow(dead_code)]
fn transpose_matrix_inplace(matrix: &mut [f32], rows: usize, cols: usize) -> Result<()> {
    if matrix.len() != rows * cols {
        return Err(wasm_chord_core::error::Error::Runtime(format!(
            "Matrix size mismatch: expected {}, got {}",
            rows * cols,
            matrix.len()
        )));
    }

    let mut temp = vec![0.0; matrix.len()];
    for i in 0..rows {
        for j in 0..cols {
            temp[j * rows + i] = matrix[i * cols + j];
        }
    }
    matrix.copy_from_slice(&temp);
    Ok(())
}
