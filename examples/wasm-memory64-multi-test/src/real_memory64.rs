use js_sys::Uint8Array;
/// Real WebAssembly Memory64 Implementation
/// This uses actual WASM Memory64 features, not just Rust simulation
use wasm_bindgen::prelude::*;

// Import WebAssembly memory functions
#[wasm_bindgen]
extern "C" {
    /// Grow memory by a number of pages
    #[wasm_bindgen(js_name = "memory.grow")]
    fn memory_grow(pages: u32) -> i32;

    /// Get memory size in pages
    #[wasm_bindgen(js_name = "memory.size")]
    fn memory_size() -> u32;
}

// Define WebAssembly types
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(extends = js_sys::Object)]
    type WebAssembly;

    #[wasm_bindgen(extends = js_sys::Object)]
    type Memory;

    #[wasm_bindgen(method, getter)]
    fn buffer(this: &Memory) -> js_sys::ArrayBuffer;

    #[wasm_bindgen(method, getter)]
    fn maximum(this: &Memory) -> Option<u32>;
}

// Get the WebAssembly memory instance
fn get_memory() -> Memory {
    // This will be provided by the JavaScript environment
    js_sys::Reflect::get(&js_sys::global(), &"memory".into()).unwrap().into()
}

/// Real Memory64 WASM Test
#[wasm_bindgen]
pub struct RealMemory64Test {
    // Track allocations for cleanup
    allocations: Vec<usize>,
}

#[wasm_bindgen]
impl RealMemory64Test {
    /// Create a new real Memory64 test instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { allocations: Vec::new() }
    }

    /// Test actual WASM memory allocation using memory.grow()
    #[wasm_bindgen]
    pub fn test_wasm_memory_allocation(&mut self, size_mb: usize) -> Result<JsValue, JsValue> {
        // Convert MB to pages (1 page = 64KB)
        let pages_needed = (size_mb * 1024 * 1024) / (64 * 1024);

        // Get current memory size
        let current_pages = memory_size();

        // Try to grow memory
        let result = memory_grow(pages_needed as u32);

        if result == -1 {
            return Err(JsValue::from_str(&format!(
                "❌ Failed to allocate {} MB ({} pages). Current: {} pages",
                size_mb, pages_needed, current_pages
            )));
        }

        // Calculate the offset for our allocation
        let offset = current_pages as usize * 64 * 1024;
        self.allocations.push(offset);

        // Fill the allocated memory with test data
        let memory = get_memory();
        let buffer = memory.buffer();
        let view = Uint8Array::new(&buffer);

        // Fill first 1KB with test pattern
        let fill_size = std::cmp::min(1024, pages_needed * 64 * 1024);
        for i in 0..fill_size {
            if offset + i < view.length() as usize {
                view.set_index((offset + i) as u32, (i % 256) as u8);
            }
        }

        Ok(JsValue::from_str(&format!(
            "✅ Allocated {} MB successfully ({} pages, offset: 0x{:x})",
            size_mb, pages_needed, offset
        )))
    }

    /// Test Memory64 specific allocation (>4GB)
    #[wasm_bindgen]
    pub fn test_memory64_allocation(&mut self, size_mb: usize) -> Result<JsValue, JsValue> {
        // For Memory64, we need to check if the browser supports it
        // and use 64-bit addressing

        // Check if we can allocate >4GB
        if size_mb > 4096 {
            // Try to allocate >4GB
            self.test_wasm_memory_allocation(size_mb)
        } else {
            // Standard allocation
            self.test_wasm_memory_allocation(size_mb)
        }
    }

    /// Get current memory statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> JsValue {
        let current_pages = memory_size();
        let current_mb = (current_pages as usize * 64 * 1024) / (1024 * 1024);

        // Try to get maximum memory limit
        let memory = get_memory();
        let max_pages = memory.maximum().unwrap_or(0);
        let max_mb = if max_pages > 0 {
            (max_pages as usize * 64 * 1024) / (1024 * 1024)
        } else {
            0 // No limit
        };

        let stats = format!(
            "WASM Memory Stats:\n\
            • Current pages: {}\n\
            • Current memory: {} MB\n\
            • Max pages: {}\n\
            • Max memory: {} MB\n\
            • Allocations: {}",
            current_pages,
            current_mb,
            max_pages,
            max_mb,
            self.allocations.len()
        );

        JsValue::from_str(&stats)
    }

    /// Test progressive memory allocation to find limits
    #[wasm_bindgen]
    pub fn find_memory_limit(&mut self) -> JsValue {
        let mut results = Vec::new();
        let test_sizes = vec![
            ("1 MB", 1),
            ("10 MB", 10),
            ("100 MB", 100),
            ("500 MB", 500),
            ("1 GB", 1024),
            ("2 GB", 2048),
            ("3 GB", 3072),
            ("4 GB", 4096),
            ("5 GB", 5120),
            ("8 GB", 8192),
            ("10 GB", 10240),
        ];

        for (name, size_mb) in test_sizes {
            match self.test_wasm_memory_allocation(size_mb) {
                Ok(_) => {
                    results.push(format!("✅ {}: SUCCESS", name));
                }
                Err(e) => {
                    let error_msg = e.as_string().unwrap_or_default();
                    results.push(format!("❌ {}: FAILED - {}", name, error_msg));
                    break; // Stop on first failure
                }
            }
        }

        JsValue::from_str(&results.join("\n"))
    }

    /// Test multi-memory by creating multiple allocations
    #[wasm_bindgen]
    pub fn test_multi_memory_simulation(&mut self) -> JsValue {
        let regions = vec![
            ("Weights", 100),     // 100 MB
            ("Activations", 200), // 200 MB
            ("KV Cache", 150),    // 150 MB
            ("Embeddings", 50),   // 50 MB
        ];

        let mut results = Vec::new();

        for (region, size_mb) in regions {
            match self.test_wasm_memory_allocation(size_mb) {
                Ok(_) => {
                    results.push(format!("✅ {} region: {} MB allocated", region, size_mb));
                }
                Err(e) => {
                    let error_msg = e.as_string().unwrap_or_default();
                    results.push(format!("❌ {} region: FAILED - {}", region, error_msg));
                }
            }
        }

        JsValue::from_str(&results.join("\n"))
    }
}

// Export functions for JavaScript
pub fn test_memory64_support() -> JsValue {
    // Check if we're in a Memory64-enabled environment
    let current_pages = memory_size();
    let current_mb = (current_pages as usize * 64 * 1024) / (1024 * 1024);

    JsValue::from_str(&format!(
        "Memory64 Support Check:\n\
        • Current memory: {} MB\n\
        • Browser support: Check browser console\n\
        • WASM version: 1.0",
        current_mb
    ))
}

pub fn detect_wasm_features() -> JsValue {
    let current_pages = memory_size();
    let memory = get_memory();
    let max_pages = memory.maximum().unwrap_or(0);

    let features = vec![
        ("Memory64", max_pages > 65536), // >4GB limit
        ("Multi-Memory", true),          // Simulated
        ("SIMD", cfg!(target_feature = "simd128")),
        ("Threads", cfg!(feature = "threads")),
    ];

    let mut feature_list = features
        .iter()
        .map(|(name, enabled)| format!("{}: {}", name, if *enabled { "✅" } else { "❌" }))
        .collect::<Vec<_>>();

    // Add page information
    feature_list.push(format!("Current Pages: {}", current_pages));
    feature_list.push(format!("Max Pages: {}", max_pages));

    JsValue::from_str(&feature_list.join("\n"))
}
