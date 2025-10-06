/// GPU backend matmul tests
/// Note: These tests require GPU hardware or WebGPU in browser
use wasm_chord_gpu::GpuBackend;

#[test]
#[ignore] // Requires GPU - run with: cargo test --package wasm-chord-gpu -- --ignored
fn test_gpu_matmul_small() {
    pollster::block_on(async {
        let gpu = GpuBackend::new().await.expect("Failed to init GPU");

        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = gpu.matmul(&a, &b, 2, 3, 2).expect("Matmul failed");

        // Expected: [22, 28, 49, 64]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 22.0).abs() < 0.001);
        assert!((result[1] - 28.0).abs() < 0.001);
        assert!((result[2] - 49.0).abs() < 0.001);
        assert!((result[3] - 64.0).abs() < 0.001);
    });
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_matmul_large() {
    pollster::block_on(async {
        let gpu = GpuBackend::new().await.expect("Failed to init GPU");

        // Test larger matrix (64x64)
        let size = 64;
        let a = vec![1.0; size * size];
        let b = vec![2.0; size * size];

        let result = gpu.matmul(&a, &b, size as u32, size as u32, size as u32)
            .expect("Matmul failed");

        // Each element should be size * 1.0 * 2.0
        assert_eq!(result.len(), size * size);
        let expected = (size as f32) * 2.0;
        for val in result {
            assert!((val - expected).abs() < 0.1, "Expected {}, got {}", expected, val);
        }
    });
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_matmul_identity() {
    pollster::block_on(async {
        let gpu = GpuBackend::new().await.expect("Failed to init GPU");

        // Test with identity matrix
        let size = 4;
        let mut identity = vec![0.0; size * size];
        for i in 0..size {
            identity[i * size + i] = 1.0;
        }

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        let result = gpu.matmul(&data, &identity, 4, 4, 4).expect("Matmul failed");

        // Should be unchanged
        for i in 0..data.len() {
            assert!((result[i] - data[i]).abs() < 0.001);
        }
    });
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_performance_matmul() {
    pollster::block_on(async {
        let gpu = GpuBackend::new().await.expect("Failed to init GPU");

        // Test performance with larger matrix
        let size = 256;
        let a = vec![1.0; size * size];
        let b = vec![2.0; size * size];

        let start = std::time::Instant::now();
        let _result = gpu.matmul(&a, &b, size as u32, size as u32, size as u32)
            .expect("Matmul failed");
        let duration = start.elapsed();

        println!("GPU matmul 256x256 took: {:?}", duration);

        // GPU should be reasonably fast
        assert!(duration.as_millis() < 1000, "GPU matmul too slow: {:?}", duration);
    });
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_error_handling() {
    pollster::block_on(async {
        let gpu = GpuBackend::new().await.expect("Failed to init GPU");

        // Test with mismatched dimensions
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];

        let result = gpu.matmul(&a, &b, 2, 2, 2);
        assert!(result.is_err(), "Should error on mismatched dimensions");
    });
}

#[test]
fn test_gpu_availability() {
    // Test that availability check doesn't crash
    let available = GpuBackend::is_available();

    // On native (non-WASM), should report availability
    #[cfg(not(target_arch = "wasm32"))]
    assert!(available, "GPU should be available on native platforms");

    // On WASM, depends on browser support
    #[cfg(target_arch = "wasm32")]
    println!("GPU available: {}", available);
}

#[test]
#[ignore] // Requires GPU
fn test_gpu_multiple_matmuls() {
    // Test multiple operations in sequence
    pollster::block_on(async {
        let gpu = GpuBackend::new().await.expect("Failed to init GPU");

        // Operation 1: Small matmul
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result1 = gpu.matmul(&a, &b, 2, 2, 2).expect("Matmul 1 failed");
        assert_eq!(result1.len(), 4);

        // Operation 2: Different size
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let result2 = gpu.matmul(&c, &d, 2, 3, 2).expect("Matmul 2 failed");
        assert_eq!(result2.len(), 4);

        // Operation 3: Larger
        let size = 32;
        let e = vec![1.0; size * size];
        let f = vec![2.0; size * size];
        let result3 = gpu.matmul(&e, &f, size as u32, size as u32, size as u32)
            .expect("Matmul 3 failed");
        assert_eq!(result3.len(), size * size);
    });
}
