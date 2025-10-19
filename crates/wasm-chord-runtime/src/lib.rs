//! wasm-chord runtime with stable ABI
//!
//! Provides both C ABI and wasm-bindgen interfaces for host integration.

mod abi;
mod chat;
mod context;
mod inference;
mod memory;
pub mod memory64;

// Host-side Memory64 runtime (production-hardened)
#[cfg(feature = "memory64-host")]
pub mod memory64_host;

// WASM-side Memory64 FFI bindings
#[cfg(all(feature = "memory64-wasm", target_arch = "wasm32"))]
pub mod memory64_ffi;

mod multi_memory;
mod sampling;
mod sharding;
pub mod streaming;
mod transformer;

#[cfg(target_arch = "wasm32")]
mod web;

pub use abi::*;
pub use chat::{ChatMessage, ChatRole, ChatTemplate};
pub use context::RuntimeContext;
pub use inference::{GenOptions, GenerationState, InferenceSession};
pub use memory::{estimate_model_memory, requires_memory64, MemoryAllocator, MemoryConfig};
pub use memory64::{get_max_memory_size, supports_memory64, WasmMemory64Allocator};

// Re-export Memory64 types (already exported from memory64 module, but keep for convenience)
#[cfg(feature = "memory64-host")]
pub use memory64::{
    LayerInfo, Memory64Runtime, Memory64State, MemoryLayout, MemoryRegion as Memory64Region,
    MemoryStats,
};

#[cfg(all(feature = "memory64-wasm", target_arch = "wasm32"))]
pub use memory64::{
    get_memory64_stats, is_memory64_enabled, load_layer, read_memory64, Memory64LayerLoader,
};

pub use multi_memory::{MemoryRegion, MemoryRegionConfig, MultiMemoryLayout};
pub use sampling::{LogitsProcessor, Sampling};
pub use sharding::{ShardConfig, ShardingManager, ShardingStrategy};
pub use transformer::{
    AttentionWeights, FFNWeights, FeedForward, GenerationConfig, KVCache, Model,
    MultiHeadAttention, TransformerConfig, TransformerLayer,
};

use wasm_chord_core::error::Error;

/// Runtime version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error codes for C ABI
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Ok = 0,
    GenericFailure = 1,
    OutOfMemory = 2,
    InvalidArgument = 3,
    ModelParseError = 4,
    BackendUnsupported = 5,
    ShaderCompileError = 6,
    IoError = 7,
}

impl From<Error> for ErrorCode {
    fn from(e: Error) -> Self {
        match e {
            Error::InvalidShape(_) | Error::UnsupportedDataType(_) => ErrorCode::InvalidArgument,
            Error::AllocationFailed(_) => ErrorCode::OutOfMemory,
            Error::ParseError(_) | Error::InvalidFormat(_) => ErrorCode::ModelParseError,
            Error::BackendError(_) => ErrorCode::BackendUnsupported,
            Error::Io(_) => ErrorCode::IoError,
            Error::Runtime(_) => ErrorCode::GenericFailure,
            Error::Unknown(_) => ErrorCode::GenericFailure,
        }
    }
}

/// Thread-local error message storage for C ABI
use std::cell::RefCell;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub(crate) fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
}

pub(crate) fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|e| e.borrow_mut().take())
}
