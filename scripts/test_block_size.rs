use wasm_chord_core::quant::BlockQ4_K;

fn main() {
    println!("BlockQ4_K size: {} bytes", std::mem::size_of::<BlockQ4_K>());
    println!("Expected: 144 bytes (2 + 2 + 12 + 128)");
}
