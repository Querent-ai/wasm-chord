//! wasm-chord runtime with stable ABI
//!
//! Provides both C ABI and wasm-bindgen interfaces for host integration.

mod abi;
mod context;
mod inference;

pub use abi::*;
pub use context::RuntimeContext;
pub use inference::InferenceSession;

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
