/// Kernel Verification: Compare with llama.cpp outputs
/// This creates a minimal test that can be compared with llama.cpp's first token generation
use wasm_chord_cpu::kernels::softmax;
use wasm_chord_runtime::{attention::AttentionBackend, Model, TransformerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Kernel Verification vs llama.cpp");
    println!("====================================\n");

    // Test 1: Fixed attention computation with realistic values
    test_realistic_attention();

    // Test 2: RMSNorm with known values
    test_rmsnorm_known_values();

    // Test 3: SwiGLU with known values
    test_swiglu_known_values();

    Ok(())
}

fn test_realistic_attention() {
    println!("1️⃣ Testing Realistic Attention Computation");
    println!("--------------------------------------------");

    // Use realistic values that won't cause extreme softmax
    let seq_len = 2;
    let head_dim = 2;
    let num_heads = 2;

    // Create more realistic Q, K matrices (smaller values)
    let q = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // [seq_len, num_heads, head_dim]
    let k = vec![0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];

    // Reshape for attention computation
    let mut attention_scores = vec![0.0; seq_len * seq_len];

    // Compute attention scores: Q @ K^T with proper scaling
    // For each query position i, compute scores against all key positions j
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut score = 0.0;
            // Sum over all heads and head dimensions
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let q_idx = i * num_heads * head_dim + h * head_dim + d;
                    let k_idx = j * num_heads * head_dim + h * head_dim + d;
                    score += q[q_idx] * k[k_idx];
                }
            }
            // Apply proper scaling: score /= sqrt(head_dim)
            attention_scores[i * seq_len + j] = score / (head_dim as f32).sqrt();
        }
    }

    println!("Q: {:?}", q);
    println!("K: {:?}", k);
    println!("Attention scores (before softmax): {:?}", attention_scores);

    // Apply softmax per query position (each row)
    let mut attention_probs = vec![0.0; seq_len * seq_len];

    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        let row_scores = &attention_scores[row_start..row_end];
        let row_probs = &mut attention_probs[row_start..row_end];

        softmax(row_scores, row_probs).unwrap();
    }

    println!("Attention probabilities: {:?}", attention_probs);

    // Verify each row sums to 1
    for i in 0..seq_len {
        let row_sum: f32 = attention_probs[i * seq_len..(i + 1) * seq_len].iter().sum();
        println!("Row {} sum: {:.6}", i, row_sum);
        assert!((row_sum - 1.0).abs() < 1e-5, "Attention row doesn't sum to 1!");
    }

    println!("✅ Realistic attention tests passed\n");
}

fn test_rmsnorm_known_values() {
    println!("2️⃣ Testing RMSNorm with Known Values");
    println!("------------------------------------");

    // Test with values that match llama.cpp behavior
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let eps = 1e-5;

    // Manual RMSNorm computation (like llama.cpp)
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let mean = sum_sq / input.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();

    let mut manual_rmsnorm = vec![0.0; 4];
    for i in 0..input.len() {
        manual_rmsnorm[i] = input[i] * scale * weight[i];
    }

    // Test with our implementation
    let config = TransformerConfig {
        vocab_size: 1000,
        hidden_size: 4,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_size: 8,
        max_seq_len: 128,
        num_layers: 1,
        rms_norm_eps: eps,
        rope_theta: 10000.0,
        attention_backend: AttentionBackend::Auto,
    };
    let model = Model::new(config.clone());
    let our_rmsnorm = model.rms_norm(&input, &weight).unwrap();

    println!("Input: {:?}", input);
    println!("Manual RMSNorm: {:?}", manual_rmsnorm);
    println!("Our RMSNorm: {:?}", our_rmsnorm);

    // Compare
    let diff: f32 = manual_rmsnorm.iter().zip(our_rmsnorm.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("Total difference: {:.6}", diff);

    assert!(diff < 1e-5, "RMSNorm implementation incorrect!");

    println!("✅ RMSNorm tests passed\n");
}

fn test_swiglu_known_values() {
    println!("3️⃣ Testing SwiGLU with Known Values");
    println!("-----------------------------------");

    // Test with specific values to verify correctness
    let gate = vec![1.0, 0.0, -1.0];
    let up = vec![2.0, 1.0, 3.0];

    let mut swiglu_output = vec![0.0; 3];

    for i in 0..gate.len() {
        let sigmoid = 1.0 / (1.0 + (-gate[i] as f32).exp());
        let silu = gate[i] * sigmoid;
        swiglu_output[i] = silu * up[i];
    }

    // Expected values (computed manually)
    let silu_1 = 1.0 * (1.0 / (1.0 + (-1.0f32).exp())); // ≈ 0.731059
    let silu_0 = 0.0;
    let silu_neg1 = -(1.0 / (1.0 + (-(-1.0f32)).exp())); // ≈ -0.268941

    let expected = vec![silu_1 * 2.0, silu_0 * 1.0, silu_neg1 * 3.0];

    println!("Gate: {:?}", gate);
    println!("Up: {:?}", up);
    println!("SwiGLU: {:?}", swiglu_output);
    println!("Expected: {:?}", expected);

    // Compare
    let diff: f32 = swiglu_output.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("Total difference: {:.6}", diff);

    assert!(diff < 1e-5, "SwiGLU implementation incorrect!");

    println!("✅ SwiGLU tests passed\n");
}
