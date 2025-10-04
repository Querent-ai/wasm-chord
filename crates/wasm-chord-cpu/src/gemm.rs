/// General Matrix Multiply (GEMM) kernels
///
/// Implements naive and blocked matrix multiplication for f32.
use wasm_chord_core::error::{Error, Result};

/// Naive matrix multiplication: C = A * B
///
/// A: [M, K]
/// B: [K, N]
/// C: [M, N]
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    if a.len() != m * k {
        return Err(Error::InvalidShape(format!(
            "Matrix A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(Error::InvalidShape(format!(
            "Matrix B size mismatch: expected {}, got {}",
            k * n,
            b.len()
        )));
    }
    if c.len() != m * n {
        return Err(Error::InvalidShape(format!(
            "Matrix C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }

    // Zero output
    c.fill(0.0);

    // Naive triple loop (can be optimized with blocking/SIMD later)
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    Ok(())
}

/// Matrix multiplication with transposed B: C = A * B^T
///
/// A: [M, K]
/// B: [N, K] (will be accessed as transposed)
/// C: [M, N]
///
/// This is more cache-friendly when B is stored row-major.
pub fn matmul_transposed(
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    if a.len() != m * k {
        return Err(Error::InvalidShape(format!(
            "Matrix A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if b_t.len() != n * k {
        return Err(Error::InvalidShape(format!(
            "Matrix B^T size mismatch: expected {}, got {}",
            n * k,
            b_t.len()
        )));
    }
    if c.len() != m * n {
        return Err(Error::InvalidShape(format!(
            "Matrix C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }

    c.fill(0.0);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                // B is stored as [N, K], so B^T[j, l] = B[j * k + l]
                sum += a[i * k + l] * b_t[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }

    Ok(())
}

/// Blocked GEMM for better cache locality (future optimization)
#[allow(dead_code)]
fn matmul_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
) -> Result<()> {
    c.fill(0.0);

    for i0 in (0..m).step_by(block_size) {
        for j0 in (0..n).step_by(block_size) {
            for l0 in (0..k).step_by(block_size) {
                let i_max = (i0 + block_size).min(m);
                let j_max = (j0 + block_size).min(n);
                let l_max = (l0 + block_size).min(k);

                for i in i0..i_max {
                    for j in j0..j_max {
                        let mut sum = c[i * n + j];
                        for l in l0..l_max {
                            sum += a[i * k + l] * b[l * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity matrix
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();

        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_matmul_simple() {
        // [2, 3] * [3, 2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];

        matmul_f32(&a, &b, &mut c, 2, 3, 2).unwrap();

        // Expected:
        // C[0,0] = 1*1 + 2*3 + 3*5 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 64
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_transposed() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b_t = vec![1.0, 3.0, 2.0, 4.0]; // [2, 2] stored as transpose
        let mut c = vec![0.0; 4];

        matmul_transposed(&a, &b_t, &mut c, 2, 2, 2).unwrap();

        // B^T means we're multiplying by [[1, 2], [3, 4]]
        // A * B = [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]]
        //       = [[7, 10], [15, 22]]
        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
    }
}
