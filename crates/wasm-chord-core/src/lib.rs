//! Core inference primitives for wasm-chord
//!
//! This crate provides the fundamental building blocks for neural network inference:
//! - Tensor types and operations
//! - Model format parsing (GGUF)
//! - Quantization/dequantization primitives
//! - Memory management abstractions

pub mod error;
pub mod formats;
pub mod memory;
pub mod quant;
pub mod tensor;

pub use error::{Error, Result};
pub use formats::gguf::{GGUFParser, ModelMeta};
pub use tensor::{DataType, Shape, Tensor, TensorDesc};

/// Core version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
