//! Core inference primitives for wasm-chord
//!
//! This crate provides the fundamental building blocks for neural network inference:
//! - Tensor types and operations
//! - Model format parsing (GGUF)
//! - Quantization/dequantization primitives
//! - Memory management abstractions

pub mod tensor;
pub mod formats;
pub mod quant;
pub mod memory;
pub mod error;

pub use error::{Error, Result};
pub use tensor::{Tensor, TensorDesc, Shape, DataType};
pub use formats::gguf::{GGUFParser, ModelMeta};

/// Core version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
