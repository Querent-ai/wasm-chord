//! CPU backend for wasm-chord
//!
//! Provides SIMD-accelerated kernels for tensor operations on CPU.

pub mod gemm;
pub mod kernels;

pub use gemm::{matmul_f32, matmul_transposed};

use wasm_chord_core::error::Result;

/// CPU backend configuration
#[derive(Debug, Clone)]
pub struct CpuBackend {
    /// Number of threads to use (0 = auto-detect)
    pub num_threads: usize,
    /// Enable SIMD optimizations
    pub use_simd: bool,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            use_simd: true,
        }
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    pub fn init(&self) -> Result<()> {
        // Initialize rayon thread pool if specified
        if self.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build_global()
                .map_err(|e| wasm_chord_core::error::Error::BackendError(e.to_string()))?;
        }
        Ok(())
    }
}
