use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wasm_chord_cpu::matmul_f32;

fn bench_gemm_small(c: &mut Criterion) {
    let m = 128;
    let k = 128;
    let n = 128;

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    c.bench_function("gemm_128x128x128", |bencher| {
        bencher.iter(|| {
            matmul_f32(black_box(&a), black_box(&b), black_box(&mut c), m, k, n).unwrap();
        });
    });
}

fn bench_gemm_medium(c: &mut Criterion) {
    let m = 512;
    let k = 512;
    let n = 512;

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    c.bench_function("gemm_512x512x512", |bencher| {
        bencher.iter(|| {
            matmul_f32(black_box(&a), black_box(&b), black_box(&mut c), m, k, n).unwrap();
        });
    });
}

criterion_group!(benches, bench_gemm_small, bench_gemm_medium);
criterion_main!(benches);
