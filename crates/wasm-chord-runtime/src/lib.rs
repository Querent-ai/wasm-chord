//! wasm-chord runtime with stable ABI
//!
//! Provides both C ABI and wasm-bindgen interfaces for host integration.

mod abi;
mod chat;
mod context;
mod inference;
mod memory;
mod memory64;
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
