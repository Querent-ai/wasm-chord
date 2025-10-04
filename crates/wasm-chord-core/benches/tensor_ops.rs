use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wasm_chord_core::tensor::{DataType, Shape, TensorDesc};

fn bench_tensor_desc_creation(c: &mut Criterion) {
    c.bench_function("tensor_desc_create", |b| {
        b.iter(|| {
            TensorDesc::new(
                black_box("test.weight".to_string()),
                black_box(DataType::F32),
                black_box(Shape::new(vec![128, 256])),
                black_box(0),
            )
            .unwrap()
        });
    });
}

fn bench_shape_validation(c: &mut Criterion) {
    let shape = Shape::new(vec![128, 256, 512]);

    c.bench_function("shape_validate", |b| {
        b.iter(|| black_box(&shape).validate().unwrap());
    });
}

criterion_group!(benches, bench_tensor_desc_creation, bench_shape_validation);
criterion_main!(benches);
