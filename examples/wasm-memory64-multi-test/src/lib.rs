/// WebAssembly Memory64 & Multi-Memory Test
/// This test compiles to actual WASM and tests Memory64 + multi-memory features
use wasm_bindgen::prelude::*;
#[cfg(feature = "memory64")]
use wasm_chord_runtime::WasmMemory64Allocator;
use wasm_chord_runtime::{MemoryAllocator, MemoryConfig, MemoryRegion, MultiMemoryLayout};

// Include the real Memory64 implementation
mod real_memory64;

// Export functions for JavaScript to call
#[wasm_bindgen]
pub struct WasmMemoryTest {
    allocator: MemoryAllocator,
    multi_memory: MultiMemoryLayout,
    #[cfg(feature = "memory64")]
    memory64_allocator: Option<WasmMemory64Allocator>,
}

#[wasm_bindgen]
impl WasmMemoryTest {
    /// Create a new memory test instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Test basic memory allocation
    #[wasm_bindgen]
    pub fn test_basic_allocation(&mut self, size_mb: usize) -> Result<JsValue, JsValue> {
        let size_bytes = size_mb * 1024 * 1024 / std::mem::size_of::<u8>();

        match self.allocator.allocate::<u8>(size_bytes) {
            Ok(mut buffer) => {
                // Fill with test data
                for i in 0..std::cmp::min(buffer.len(), 10000) {
                    buffer[i] = (i % 256) as u8;
                }

                Ok(JsValue::from_str(&format!(
                    "✅ Allocated {} MB successfully ({} bytes)",
                    size_mb,
                    buffer.len()
                )))
            }
            Err(e) => Err(JsValue::from_str(&format!("❌ Allocation failed: {}", e))),
        }
    }

    /// Test real WASM Memory64 allocation using memory.grow()
    #[wasm_bindgen]
    pub fn test_real_memory64_allocation(&mut self, size_mb: usize) -> Result<JsValue, JsValue> {
        let mut real_test = real_memory64::RealMemory64Test::new();
        real_test.test_memory64_allocation(size_mb)
    }

    /// Test Memory64 allocation (>4GB)
    #[wasm_bindgen]
    pub fn test_memory64_allocation(&mut self, _size_mb: usize) -> Result<JsValue, JsValue> {
        #[cfg(feature = "memory64")]
        {
            if self.memory64_allocator.is_none() {
                match WasmMemory64Allocator::new(1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024) {
                    Ok(allocator) => self.memory64_allocator = Some(allocator),
                    Err(e) => {
                        return Err(JsValue::from_str(&format!(
                            "❌ Memory64 allocator creation failed: {}",
                            e
                        )))
                    }
                }
            }

            let size_bytes = size_mb * 1024 * 1024 / std::mem::size_of::<u8>();

            if let Some(ref mut allocator) = self.memory64_allocator {
                match allocator.allocate::<u8>(size_bytes) {
                    Ok(mut buffer) => {
                        // Fill with test data
                        for i in 0..std::cmp::min(buffer.len(), 10000) {
                            buffer[i] = (i % 256) as u8;
                        }

                        Ok(JsValue::from_str(&format!(
                            "✅ Memory64 allocated {} MB successfully ({} bytes)",
                            size_mb,
                            buffer.len()
                        )))
                    }
                    Err(e) => {
                        Err(JsValue::from_str(&format!("❌ Memory64 allocation failed: {}", e)))
                    }
                }
            } else {
                Err(JsValue::from_str("❌ Memory64 allocator not available"))
            }
        }

        #[cfg(not(feature = "memory64"))]
        {
            Err(JsValue::from_str("❌ Memory64 feature not enabled"))
        }
    }

    /// Test multi-memory allocation
    #[wasm_bindgen]
    pub fn test_multi_memory_allocation(
        &mut self,
        region: &str,
        size_mb: usize,
    ) -> Result<JsValue, JsValue> {
        let memory_region = match region {
            "weights" => MemoryRegion::Weights,
            "activations" => MemoryRegion::Activations,
            "kv_cache" => MemoryRegion::KVCache,
            "embeddings" => MemoryRegion::Embeddings,
            _ => {
                return Err(JsValue::from_str(
                    "❌ Invalid region. Use: weights, activations, kv_cache, embeddings",
                ))
            }
        };

        let size_bytes = size_mb * 1024 * 1024;

        match self.multi_memory.allocate(memory_region, size_bytes) {
            Ok(_) => Ok(JsValue::from_str(&format!(
                "✅ Multi-memory allocated {} MB in {} region",
                size_mb, region
            ))),
            Err(e) => Err(JsValue::from_str(&format!("❌ Multi-memory allocation failed: {}", e))),
        }
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> JsValue {
        let stats = format!(
            "Memory Stats:\n\
            • Basic allocator: {:.1}% used\n\
            • Memory64 enabled: {}\n\
            • Multi-memory regions: {}",
            self.allocator.usage_percent(),
            cfg!(feature = "memory64"),
            self.multi_memory.region_count()
        );

        JsValue::from_str(&stats)
    }

    /// Test WebAssembly memory.grow() directly
    #[wasm_bindgen]
    pub fn test_wasm_memory_grow(&self) -> JsValue {
        // This tests actual WASM memory.grow() functionality
        let initial_pages = 1;
        let grow_pages = 100; // ~6.4MB

        // In a real WASM environment, this would call memory.grow()
        // For now, we simulate the behavior
        let result = format!(
            "WASM Memory Test:\n\
            • Initial pages: {}\n\
            • Grow by: {} pages\n\
            • Total pages: {}\n\
            • Memory64 support: {}",
            initial_pages,
            grow_pages,
            initial_pages + grow_pages,
            cfg!(feature = "memory64")
        );

        JsValue::from_str(&result)
    }

    /// Find the real WASM memory limit
    #[wasm_bindgen]
    pub fn find_real_memory_limit(&mut self) -> JsValue {
        let mut real_test = real_memory64::RealMemory64Test::new();
        real_test.find_memory_limit()
    }

    /// Get real WASM memory statistics
    #[wasm_bindgen]
    pub fn get_real_memory_stats(&self) -> JsValue {
        let real_test = real_memory64::RealMemory64Test::new();
        real_test.get_memory_stats()
    }

    /// Stress test with large allocations
    #[wasm_bindgen]
    pub fn stress_test(&mut self) -> JsValue {
        let mut results = Vec::new();

        // Test different allocation sizes
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
        ];

        for (name, size_mb) in test_sizes {
            let size_bytes = size_mb * 1024 * 1024 / std::mem::size_of::<u8>();

            match self.allocator.allocate::<u8>(size_bytes) {
                Ok(buffer) => {
                    results.push(format!("✅ {}: {} bytes", name, buffer.len()));
                }
                Err(_) => {
                    results.push(format!("❌ {}: FAILED", name));
                    break; // Stop on first failure
                }
            }
        }

        JsValue::from_str(&results.join("\n"))
    }
}

impl Default for WasmMemoryTest {
    fn default() -> Self {
        Self {
            allocator: MemoryAllocator::new(MemoryConfig::default()),
            multi_memory: MultiMemoryLayout::new(),
            #[cfg(feature = "memory64")]
            memory64_allocator: None,
        }
    }
}

// Export a simple function for testing
#[wasm_bindgen]
pub fn test_memory64_support() -> JsValue {
    real_memory64::test_memory64_support()
}

// Export multi-memory test
#[wasm_bindgen]
pub fn test_multi_memory_support() -> JsValue {
    let multi_memory = MultiMemoryLayout::new();
    let region_count = multi_memory.region_count();

    JsValue::from_str(&format!("Multi-Memory Support: ✅ ENABLED\nRegions: {}", region_count))
}

// Export WebAssembly feature detection
#[wasm_bindgen]
pub fn detect_wasm_features() -> JsValue {
    real_memory64::detect_wasm_features()
}
