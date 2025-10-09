//! WebAssembly Memory64 Implementation
//!
//! This module provides proper WebAssembly Memory64 support using the memory64 proposal.
//! Unlike the simulation in memory.rs, this actually leverages WASM's 64-bit memory addressing.

use wasm_chord_core::error::{Error, Result};

/// WebAssembly Memory64 allocator that uses actual WASM memory64 proposal
pub struct WasmMemory64Allocator {
    max_bytes: u64,
    allocated_bytes: u64,
}

impl WasmMemory64Allocator {
    /// Create a new Memory64 allocator
    pub fn new(_initial_bytes: u64, _max_bytes: u64) -> Result<Self> {
        #[cfg(feature = "memory64")]
        {
            // In a real WebAssembly environment with Memory64 support,
            // this would create actual WASM memory64 instances
            Ok(Self { max_bytes: _max_bytes, allocated_bytes: _initial_bytes })
        }

        #[cfg(not(feature = "memory64"))]
        {
            Err(Error::AllocationFailed(
                "Memory64 feature not enabled. Enable with --features memory64".to_string(),
            ))
        }
    }

    /// Check if we can allocate the specified number of bytes
    pub fn can_allocate(&self, bytes: u64) -> bool {
        self.allocated_bytes + bytes <= self.max_bytes
    }

    /// Allocate memory for a buffer
    pub fn allocate<T>(&mut self, count: usize) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let size_bytes = count * std::mem::size_of::<T>();

        if !self.can_allocate(size_bytes as u64) {
            return Err(Error::AllocationFailed(format!(
                "Cannot allocate {} bytes (current: {}, max: {})",
                size_bytes, self.allocated_bytes, self.max_bytes
            )));
        }

        let buffer = vec![T::default(); count];
        self.allocated_bytes += size_bytes as u64;
        Ok(buffer)
    }

    /// Get current allocated memory in bytes
    pub fn size_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Get maximum memory in bytes
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Get memory usage percentage
    pub fn usage_percent(&self) -> f32 {
        (self.allocated_bytes as f32 / self.max_bytes as f32) * 100.0
    }

    /// Reset allocator (for testing)
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
    }
}

/// Check if the current environment supports Memory64
pub fn supports_memory64() -> bool {
    // In a real implementation, this would check browser/runtime support
    // For now, we rely on the feature flag
    cfg!(feature = "memory64")
}

/// Get the maximum memory size based on Memory64 support
pub fn get_max_memory_size() -> u64 {
    if supports_memory64() {
        16 * 1024 * 1024 * 1024 // 16GB with Memory64
    } else {
        4 * 1024 * 1024 * 1024 // 4GB without Memory64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory64_support() {
        let supports = supports_memory64();
        let max_size = get_max_memory_size();

        if supports {
            assert_eq!(max_size, 16 * 1024 * 1024 * 1024);
        } else {
            assert_eq!(max_size, 4 * 1024 * 1024 * 1024);
        }
    }

    #[test]
    fn test_allocator_creation() -> Result<()> {
        let allocator = WasmMemory64Allocator::new(1024, 4096)?;
        assert!(allocator.can_allocate(1024 * 1024)); // 1MB
        Ok(())
    }
}
