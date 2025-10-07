use wasm_chord_cpu::gemm::matmul_f32;

fn main() {
    // Test case: [1, 3] @ [3, 2] = [1, 2]
    let a = vec![1.0, 2.0, 3.0];  // [1, 3]
    let b = vec![
        1.0, 2.0,  // [3, 2] - first column
        3.0, 4.0,  // second column
        5.0, 6.0,  // third column
    ];
    let mut c = vec![0.0; 2];  // [1, 2]
    
    matmul_f32(&a, &b, &mut c, 1, 3, 2);
    
    println!("A: {:?}", a);
    println!("B: {:?}", b);
    println!("C: {:?}", c);
    println!("Expected: [22.0, 28.0] (1*1+2*3+3*5, 1*2+2*4+3*6)");
    
    // Test case: [2, 3] @ [3, 2] = [2, 2]
    let a2 = vec![
        1.0, 2.0, 3.0,  // row 0
        4.0, 5.0, 6.0,  // row 1
    ];
    let mut c2 = vec![0.0; 4];  // [2, 2]
    
    matmul_f32(&a2, &b, &mut c2, 2, 3, 2);
    
    println!("\nA2: {:?}", a2);
    println!("B: {:?}", b);
    println!("C2: {:?}", c2);
    println!("Expected: [22.0, 28.0, 49.0, 64.0]");
}
