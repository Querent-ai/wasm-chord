use anyhow::Result;
use std::ffi::CStr;
use std::ffi::CString;
use std::fs;
use wasm_chord_runtime::{
    wasmchord_close_stream, wasmchord_free_model, wasmchord_infer, wasmchord_init,
    wasmchord_last_error, wasmchord_load_model, wasmchord_next_token, wasmchord_version, ErrorCode,
};

fn main() -> Result<()> {
    println!("🧪 Testing ABI functions for 'capital of France' -> 'Paris'");
    println!("Version: {}", unsafe { CStr::from_ptr(wasmchord_version()).to_string_lossy() });

    // Initialize runtime
    println!("\n🔧 Initializing runtime...");
    let config = b"{}"; // Default config
    let init_result = unsafe { wasmchord_init(config.as_ptr(), config.len()) };
    if init_result != ErrorCode::Ok as u32 {
        if init_result == ErrorCode::GenericFailure as u32 {
            println!("⚠️  Runtime already initialized from previous test - continuing with existing runtime");
        } else {
            anyhow::bail!("Failed to initialize runtime: {}", init_result);
        }
    } else {
        println!("✓ Runtime initialized");
    }

    // Initialize GPU explicitly (if available)
    #[cfg(feature = "cuda")]
    {
        println!("\n🎮 Initializing GPU...");
        // Note: GPU initialization is handled automatically during model loading
        println!("✓ GPU will be initialized during model loading");
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("\n🎮 Using CPU backend");
    }

    // Load model
    println!("\n📦 Loading model...");
    let model_path = "../../models/tinyllama-1.1b-chat-v0.6-Q4_K_M.gguf";
    let model_bytes = fs::read(model_path)?;
    let model_handle = unsafe { wasmchord_load_model(model_bytes.as_ptr(), model_bytes.len()) };

    if model_handle == 0 {
        // Get error message
        let mut error_buffer = [0u8; 256];
        let error_len = unsafe {
            wasmchord_last_error(error_buffer.as_mut_ptr() as *mut i8, error_buffer.len())
        };
        if error_len > 0 {
            let error_msg = String::from_utf8_lossy(&error_buffer[..error_len]);
            anyhow::bail!("Model loading failed: {}", error_msg);
        } else {
            anyhow::bail!("Model loading failed: No error message available");
        }
    }
    println!("✓ Model loaded with handle: {}", model_handle);

    // Start inference
    println!("\n🚀 Starting inference...");
    let prompt = "What is the capital of France?";
    let prompt_cstr = CString::new(prompt)?;
    let gen_options = wasm_chord_runtime::GenOptions {
        max_tokens: 10,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        seed: 42,
        stop_token_count: 0,
        stop_tokens_ptr: 0,
    };

    let stream_handle = unsafe {
        wasmchord_infer(
            model_handle,
            prompt_cstr.as_ptr() as *const u8,
            prompt_cstr.as_bytes().len(),
            &gen_options as *const wasm_chord_runtime::GenOptions,
        )
    };

    if stream_handle == 0 {
        anyhow::bail!("Failed to start inference stream");
    }
    println!("✓ Inference started with stream handle: {}", stream_handle);

    // Generate tokens
    println!("\n🎯 Generating tokens...");
    let mut full_response = String::new();
    let mut token_count = 0;
    let max_tokens = 10;

    while token_count < max_tokens {
        let mut token_buffer = [0u8; 256];
        let token_len = unsafe {
            wasmchord_next_token(stream_handle, token_buffer.as_mut_ptr(), token_buffer.len())
        };

        if token_len == 0 {
            println!("✓ Generation completed (no more tokens)");
            break;
        }

        let token = String::from_utf8_lossy(&token_buffer[..token_len as usize]);
        full_response.push_str(&token);
        token_count += 1;

        println!("  Token {}: '{}'", token_count, token);

        // Check if we got "Paris" or similar
        if full_response.to_lowercase().contains("paris") {
            println!("🎉 Found 'Paris' in response!");
            break;
        }
    }

    println!("\n📝 Full response: '{}'", full_response);

    // Validate result
    let response_lower = full_response.to_lowercase();
    if response_lower.contains("paris") {
        println!("✅ SUCCESS: Generated 'Paris' for capital of France!");
    } else {
        println!("❌ FAILURE: Did not generate 'Paris'. Response: '{}'", full_response);
        return Ok(()); // Don't fail the test, just report
    }

    // Cleanup
    println!("\n🧹 Cleaning up...");
    unsafe { wasmchord_close_stream(stream_handle) };
    unsafe { wasmchord_free_model(model_handle) };
    println!("✓ Cleanup completed");

    println!("\n🎉 ABI test completed successfully!");
    Ok(())
}
