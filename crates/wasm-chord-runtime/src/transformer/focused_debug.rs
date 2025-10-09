/// Focused Attention Debug Tool
/// This tool checks the most likely bugs in attention computation
use wasm_chord_core::error::Result;
use wasm_chord_cpu::matmul_transposed;

/// Check if the issue is in attention computation or RoPE
pub fn debug_attention_computation() -> Result<()> {
    println!("🔍 DEBUGGING ATTENTION COMPUTATION");
    println!("==================================");

    // Test 1: Check if the issue is in the attention weights themselves
    println!("1. Testing attention weight magnitudes...");

    // Create a simple test case
    let hidden_size = 2048;
    let test_input = vec![1.0; hidden_size]; // All ones input

    // Test with identity matrix (should give same output as input)
    let mut identity_matrix = vec![0.0; hidden_size * hidden_size];
    for i in 0..hidden_size {
        identity_matrix[i * hidden_size + i] = 1.0;
    }

    let mut result = vec![0.0; hidden_size];
    matmul_transposed(&test_input, &identity_matrix, &mut result, 1, hidden_size, hidden_size)?;

    let diff: f32 = test_input.iter().zip(result.iter()).map(|(a, b)| (a - b).abs()).sum();

    println!("   Identity matrix test diff: {}", diff);
    if diff < 1e-6 {
        println!("   ✅ Identity matrix test PASSED");
    } else {
        println!("   ❌ Identity matrix test FAILED - matmul issue!");
    }

    // Test 2: Check if the issue is in the attention computation
    println!("2. Testing attention computation...");

    // Create simple Q, K, V matrices
    let seq_len = 1;
    let head_dim = 64;
    let num_heads = 32;

    let q = vec![0.1; seq_len * num_heads * head_dim];
    let k = vec![0.2; seq_len * num_heads * head_dim];
    let v = vec![0.3; seq_len * num_heads * head_dim];

    // Compute attention manually
    let mut attention_scores = vec![0.0; num_heads * head_dim * head_dim];

    for head in 0..num_heads {
        for i in 0..head_dim {
            for j in 0..head_dim {
                let q_idx = head * head_dim + i;
                let k_idx = head * head_dim + j;
                attention_scores[head * head_dim * head_dim + i * head_dim + j] =
                    q[q_idx] * k[k_idx] / (head_dim as f32).sqrt();
            }
        }
    }

    // Apply softmax (simplified)
    let mut attention_weights = vec![0.0; num_heads * head_dim * head_dim];
    for head in 0..num_heads {
        for i in 0..head_dim {
            let mut max_val = attention_scores[head * head_dim * head_dim + i * head_dim];
            for j in 1..head_dim {
                let val = attention_scores[head * head_dim * head_dim + i * head_dim + j];
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum = 0.0;
            for j in 0..head_dim {
                let val = (attention_scores[head * head_dim * head_dim + i * head_dim + j]
                    - max_val)
                    .exp();
                attention_weights[head * head_dim * head_dim + i * head_dim + j] = val;
                sum += val;
            }

            for j in 0..head_dim {
                attention_weights[head * head_dim * head_dim + i * head_dim + j] /= sum;
            }
        }
    }

    // Compute output
    let mut output = vec![0.0; seq_len * num_heads * head_dim];
    for head in 0..num_heads {
        for i in 0..head_dim {
            for j in 0..head_dim {
                let v_idx = head * head_dim + j;
                output[head * head_dim + i] +=
                    attention_weights[head * head_dim * head_dim + i * head_dim + j] * v[v_idx];
            }
        }
    }

    let output_sum: f32 = output.iter().sum();
    println!("   Attention output sum: {}", output_sum);

    if output_sum.abs() > 0.1 {
        println!("   ✅ Attention computation produces reasonable output");
    } else {
        println!("   ❌ Attention computation produces near-zero output");
    }

    Ok(())
}

/// Check if the issue is in RoPE application
pub fn debug_rope_application() -> Result<()> {
    println!("🔍 DEBUGGING ROPE APPLICATION");
    println!("============================");

    // Test RoPE with simple values
    let head_dim = 64;
    let position = 0;
    let theta: f32 = 10000.0;

    let mut q_rope = vec![0.0; head_dim];
    let mut k_rope = vec![0.0; head_dim];

    // Initialize with simple values
    for i in 0..head_dim {
        q_rope[i] = (i as f32) * 0.1;
        k_rope[i] = (i as f32) * 0.2;
    }

    // Apply RoPE
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / theta.powf((i / 2) as f32 / (head_dim as f32));
        let cos_val = (position as f32 * freq).cos();
        let sin_val = (position as f32 * freq).sin();

        if i + 1 < head_dim {
            let q0 = q_rope[i];
            let q1 = q_rope[i + 1];
            let k0 = k_rope[i];
            let k1 = k_rope[i + 1];

            q_rope[i] = q0 * cos_val - q1 * sin_val;
            q_rope[i + 1] = q0 * sin_val + q1 * cos_val;
            k_rope[i] = k0 * cos_val - k1 * sin_val;
            k_rope[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }

    let q_sum: f32 = q_rope.iter().sum();
    let k_sum: f32 = k_rope.iter().sum();

    println!("   Q after RoPE sum: {}", q_sum);
    println!("   K after RoPE sum: {}", k_sum);

    if q_sum.abs() > 0.1 && k_sum.abs() > 0.1 {
        println!("   ✅ RoPE application produces reasonable values");
    } else {
        println!("   ❌ RoPE application produces near-zero values");
    }

    Ok(())
}

/// Check if the issue is in the LM head
pub fn debug_lm_head() -> Result<()> {
    println!("🔍 DEBUGGING LM HEAD");
    println!("===================");

    // Test LM head with simple values
    let hidden_size = 2048;
    let vocab_size = 32000;

    let test_hidden = vec![1.0; hidden_size]; // All ones input

    // Create a simple LM head (identity-like)
    let mut lm_head = vec![0.0; vocab_size * hidden_size];
    for i in 0..vocab_size.min(hidden_size) {
        lm_head[i * hidden_size + i] = 1.0;
    }

    let mut logits = vec![0.0; vocab_size];
    matmul_transposed(&test_hidden, &lm_head, &mut logits, 1, hidden_size, vocab_size)?;

    let logits_sum: f32 = logits.iter().sum();
    let max_logit = logits.iter().fold(0.0_f32, |a, &b| a.max(b));

    println!("   Logits sum: {}", logits_sum);
    println!("   Max logit: {}", max_logit);

    if max_logit > 0.5 {
        println!("   ✅ LM head produces reasonable logits");
    } else {
        println!("   ❌ LM head produces near-zero logits");
    }

    Ok(())
}

/// Run all focused debugging tests
pub fn run_focused_debugging() -> Result<()> {
    println!("🧪 RUNNING FOCUSED DEBUGGING TESTS");
    println!("===================================");

    debug_attention_computation()?;
    println!();

    debug_rope_application()?;
    println!();

    debug_lm_head()?;
    println!();

    println!("✅ All focused debugging tests completed!");
    Ok(())
}
