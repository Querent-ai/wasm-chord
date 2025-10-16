//! Simple Memory64 Test - WASM Library
//!
//! This module tests allocation and access of memory beyond the 4GB WASM limit.
//! When run with Wasmtime's Memory64 API, this should successfully allocate
//! and access 6GB+ of memory.

#![cfg(target_arch = "wasm32")]

/// Test: Try to allocate a large amount of memory (in GB)
///
/// Returns:
/// - 1 if allocation and access succeeded
/// - 0 if allocation failed
#[no_mangle]
pub extern "C" fn test_large_allocation(size_gb: u32) -> u32 {
    let size_bytes = (size_gb as usize) * 1024 * 1024 * 1024;

    // Try to allocate using Vec
    let mut buffer: Vec<u8> = Vec::new();

    // Try to reserve capacity
    if buffer.try_reserve(size_bytes).is_err() {
        return 0; // Allocation failed
    }

    // Resize to actually allocate the memory
    buffer.resize(size_bytes, 0);

    // Write to first and last byte
    buffer[0] = 42;
    buffer[size_bytes - 1] = 99;

    // Verify writes
    if buffer[0] == 42 && buffer[size_bytes - 1] == 99 {
        1 // Success!
    } else {
        0 // Verification failed
    }
}

/// Get current WASM memory size in bytes
#[no_mangle]
pub extern "C" fn get_memory_size() -> i64 {
    let pages = unsafe { core::arch::wasm32::memory_size(0) };
    (pages * 65536) as i64 // pages * 64KB
}

/// Simple test: Allocate 1MB and verify
#[no_mangle]
pub extern "C" fn test_small_allocation() -> u32 {
    let size = 1024 * 1024; // 1MB
    let mut buffer = Vec::<u8>::with_capacity(size);
    buffer.resize(size, 0);

    buffer[0] = 123;
    buffer[size - 1] = 231;

    if buffer[0] == 123 && buffer[size - 1] == 231 {
        1
    } else {
        0
    }
}
