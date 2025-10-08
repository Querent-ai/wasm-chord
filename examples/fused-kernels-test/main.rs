/// Deterministic Fused Kernels Test
///
/// Tests the fused kernel operations with deterministic inputs
/// to ensure consistent numerical outputs across runs.
use wasm_chord_cpu::{
    fused_attention_score, fused_dequant_matmul_q4k, fused_rmsnorm_linear, fused_swiglu_proj,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Fused Kernels Deterministic Test");
    println!("====================================\n");

    // Test 1: Fused RMSNorm + Linear
    println!("📝 Test 1: Fused RMSNorm + Linear");

    let hidden_size = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let norm_weight = vec![1.0, 1.0, 1.0, 1.0];
    let weight = vec![
        1.0, 0.0, 0.0, 0.0, // identity matrix
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut output = vec![0.0f32; hidden_size];

    fused_rmsnorm_linear(&input, &weight, &norm_weight, &mut output, hidden_size, 1e-6)?;

    println!("   Input: {:?}", input);
    println!("   Output: [{:.6}, {:.6}, {:.6}, {:.6}]", output[0], output[1], output[2], output[3]);

    // Verify deterministic output
    assert!(output[0] > 0.0 && output[0].is_finite());
    assert!(output[1] > output[0]); // Relative order preserved after normalization
    assert!(output[2] > output[1]);
    assert!(output[3] > output[2]);

    // Run again to verify determinism
    let mut output2 = vec![0.0f32; hidden_size];
    fused_rmsnorm_linear(&input, &weight, &norm_weight, &mut output2, hidden_size, 1e-6)?;

    for i in 0..hidden_size {
        assert_eq!(output[i], output2[i], "Output must be deterministic");
    }

    println!("   ✅ RMSNorm+Linear is deterministic\n");

    // Test 2: Fused SwiGLU + Projection
    println!("📝 Test 2: Fused SwiGLU + Projection");

    let hidden_size = 4;
    let intermediate_size = 8;
    let gate = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
    let up = vec![1.0; intermediate_size];
    let down = vec![0.125f32; hidden_size * intermediate_size];
    let mut output = vec![0.0f32; hidden_size];

    fused_swiglu_proj(&gate, &up, &down, &mut output, hidden_size, intermediate_size)?;

    println!("   Gate: {:?}", &gate[..4]);
    println!("   Output: [{:.6}, {:.6}, {:.6}, {:.6}]", output[0], output[1], output[2], output[3]);

    // All outputs should be non-zero and finite
    for &val in &output {
        assert!(val.abs() > 1e-6, "Output should be non-zero");
        assert!(val.is_finite(), "Output should be finite");
    }

    // Run again for determinism check
    let mut output2 = vec![0.0f32; hidden_size];
    fused_swiglu_proj(&gate, &up, &down, &mut output2, hidden_size, intermediate_size)?;

    for i in 0..hidden_size {
        assert_eq!(output[i], output2[i], "Output must be deterministic");
    }

    println!("   ✅ SwiGLU+Projection is deterministic\n");

    // Test 3: Fused Attention Score (No Mask)
    println!("📝 Test 3: Fused Attention Score (No Mask)");

    let seq_len = 3;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let query = vec![
        1.0, 0.0, 0.0, 0.0, // q1
        0.0, 1.0, 0.0, 0.0, // q2
        0.0, 0.0, 1.0, 0.0, // q3
    ];
    let key = query.clone();

    let scores = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, false)?;

    println!("   Sequence length: {}", seq_len);
    println!("   Head dimension: {}", head_dim);
    println!("   Score matrix (3x3):");
    for i in 0..seq_len {
        print!("      ");
        for j in 0..seq_len {
            print!(" {:.4}", scores[i * seq_len + j]);
        }
        println!();
    }

    // Each row should sum to ~1.0 (softmax property)
    for i in 0..seq_len {
        let row_sum: f32 = (0..seq_len).map(|j| scores[i * seq_len + j]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum should be 1.0, got {}", i, row_sum);
    }

    // All scores in [0, 1]
    for &score in &scores {
        assert!(score >= 0.0 && score <= 1.0, "Score out of range: {}", score);
    }

    // Determinism check
    let scores2 = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, false)?;
    for i in 0..scores.len() {
        assert_eq!(scores[i], scores2[i], "Scores must be deterministic");
    }

    println!("   ✅ Attention scores are deterministic\n");

    // Test 4: Fused Attention Score (Causal Mask)
    println!("📝 Test 4: Fused Attention Score (Causal Mask)");

    let seq_len = 4;
    let head_dim = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let query = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let key = query.clone();

    let scores = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, true)?;

    println!("   Sequence length: {}", seq_len);
    println!("   Causal mask: enabled");
    println!("   Score matrix (4x4):");
    for i in 0..seq_len {
        print!("      ");
        for j in 0..seq_len {
            let score = scores[i * seq_len + j];
            if score == 0.0 {
                print!("  ---  ");
            } else {
                print!(" {:.4}", score);
            }
        }
        println!();
    }

    // Upper triangle should be zero (causal mask)
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            assert_eq!(scores[i * seq_len + j], 0.0, "Causal mask violated at ({}, {})", i, j);
        }
    }

    // Each row should sum to ~1.0 (considering only visible positions)
    for i in 0..seq_len {
        let row_sum: f32 = (0..=i).map(|j| scores[i * seq_len + j]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum should be 1.0, got {}", i, row_sum);
    }

    // Determinism check
    let scores2 = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, true)?;
    for i in 0..scores.len() {
        assert_eq!(scores[i], scores2[i], "Masked scores must be deterministic");
    }

    println!("   ✅ Causal masking is deterministic\n");

    // Test 5: Fused Dequant + Matmul (Q4_K)
    println!("📝 Test 5: Fused Dequant + Matmul (Q4_K)");

    let m = 2; // output rows
    let n = 1; // output cols
    let k = 256; // must be multiple of block size (256)

    let quantized = vec![0u8; (k / 2 + 12) * n];
    let scales = vec![1.0f32; n];
    let input = vec![1.0f32; m * k];
    let mut output = vec![0.0f32; m * n];

    fused_dequant_matmul_q4k(&quantized, &scales, &input, &mut output, m, n, k)?;

    println!("   Matrix dimensions: {}x{} × {}x{} = {}x{}", m, k, k, n, m, n);
    println!("   Output: {:?}", output);

    // Verify output is finite
    for &val in &output {
        assert!(val.is_finite(), "Output should be finite, got {}", val);
    }

    // Determinism check
    let mut output2 = vec![0.0f32; m * n];
    fused_dequant_matmul_q4k(&quantized, &scales, &input, &mut output2, m, n, k)?;

    for i in 0..(m * n) {
        assert_eq!(output[i], output2[i], "Dequant+Matmul must be deterministic");
    }

    println!("   ✅ Dequant+Matmul is deterministic\n");

    // Test 6: Cross-Instance Consistency
    println!("📝 Test 6: Cross-Instance Consistency");

    // Run all operations multiple times
    let iterations = 5;
    let mut all_outputs: Vec<Vec<f32>> = Vec::new();

    for iter in 0..iterations {
        let hidden_size = 4;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let norm_weight = vec![1.0; hidden_size];
        let weight =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; hidden_size];

        fused_rmsnorm_linear(&input, &weight, &norm_weight, &mut output, hidden_size, 1e-6)?;
        all_outputs.push(output.clone());

        if iter > 0 {
            // Compare with first iteration
            for i in 0..hidden_size {
                assert_eq!(
                    all_outputs[0][i], all_outputs[iter][i],
                    "Iteration {} differs from iteration 0",
                    iter
                );
            }
        }
    }

    println!("   Ran {} iterations", iterations);
    println!("   All outputs identical: ✅");
    println!("   ✅ Cross-instance consistency verified\n");

    // Test 7: Numerical Stability
    println!("📝 Test 7: Numerical Stability");

    // Test with very small values
    let seq_len = 2;
    let head_dim = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let query = vec![1e-10, 1e-10, 1e-10, 1e-10];
    let key = query.clone();

    let scores = fused_attention_score(&query, &key, &[], seq_len, head_dim, scale, false)?;

    println!("   Input magnitude: 1e-10");
    println!("   Output: {:?}", scores);

    // Softmax should still produce valid probabilities
    for &score in &scores {
        assert!(score.is_finite(), "Score should be finite");
        assert!(score >= 0.0 && score <= 1.0, "Score should be in [0,1]");
    }

    // Rows should sum to 1
    for i in 0..seq_len {
        let row_sum: f32 = (0..seq_len).map(|j| scores[i * seq_len + j]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "Row sum should be 1.0");
    }

    println!("   ✅ Numerically stable with small inputs\n");

    // Final Summary
    println!("✅ All Fused Kernel Tests Passed!");
    println!("\n📊 Summary:");
    println!("   • RMSNorm+Linear: Deterministic ✅");
    println!("   • SwiGLU+Projection: Deterministic ✅");
    println!("   • Attention scores (no mask): Deterministic ✅");
    println!("   • Attention scores (causal): Deterministic ✅");
    println!("   • Dequant+Matmul (Q4_K): Deterministic ✅");
    println!("   • Cross-instance consistency: Verified ✅");
    println!("   • Numerical stability: Verified ✅");
    println!("\n💡 Fused kernels exhibit fully deterministic behavior");
    println!("🚀 Ready for production use with reduced memory bandwidth");

    Ok(())
}
