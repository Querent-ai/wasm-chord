//! Tensor loading and management from GGUF files
//!
//! Supports lazy loading, caching, and efficient memory management.

use crate::error::{Error, Result};
use crate::formats::gguf::GGUFParser;
use crate::quant::{
    dequantize_q4_0, dequantize_q6_k, dequantize_q8_0, BlockQ4_0, BlockQ6_K, BlockQ8_0,
};
use crate::tensor::{DataType, TensorDesc};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// Tensor loader with lazy loading and caching
pub struct TensorLoader {
    /// Loaded tensors cache (name -> tensor data)
    cache: HashMap<String, Vec<f32>>,
    /// Tensor metadata (name -> descriptor)
    metadata: HashMap<String, TensorMetadata>,
    /// Base offset for tensor data in file
    data_offset: u64,
}

/// Metadata about a tensor's location and format
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor descriptor (shape, dtype)
    pub desc: TensorDesc,
    /// Byte offset from data_offset
    pub offset: u64,
    /// Size in bytes
    pub size_bytes: usize,
}

impl TensorLoader {
    /// Create a new tensor loader
    pub fn new(data_offset: u64) -> Self {
        Self { cache: HashMap::new(), metadata: HashMap::new(), data_offset }
    }

    /// Register a tensor from GGUF metadata
    pub fn register_tensor(&mut self, name: String, desc: TensorDesc, offset: u64) {
        let metadata = TensorMetadata { desc: desc.clone(), offset, size_bytes: desc.size_bytes };

        self.metadata.insert(name, metadata);
    }

    /// Load a tensor by name (with caching)
    pub fn load_tensor<R: Read + Seek>(
        &mut self,
        name: &str,
        parser: &mut GGUFParser<R>,
    ) -> Result<&[f32]> {
        // Check cache first
        if self.cache.contains_key(name) {
            return Ok(&self.cache[name]);
        }

        // Get metadata
        let metadata = self
            .metadata
            .get(name)
            .ok_or_else(|| Error::Unknown(format!("Tensor not found: {}", name)))?
            .clone();

        // Load raw data
        let absolute_offset = self.data_offset + metadata.offset;
        let raw_data = parser.read_tensor_data(absolute_offset, metadata.size_bytes)?;

        // Dequantize based on dtype
        let f32_data = match metadata.desc.dtype {
            DataType::F32 => {
                // Already float32, just reinterpret bytes
                let mut result = Vec::with_capacity(metadata.desc.element_count());
                for chunk in raw_data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    result.push(f32::from_le_bytes(bytes));
                }
                result
            }
            DataType::F16 => {
                // F16 -> F32 conversion
                let mut result = Vec::with_capacity(metadata.desc.element_count());
                for chunk in raw_data.chunks_exact(2) {
                    let bytes = [chunk[0], chunk[1]];
                    let f16_bits = u16::from_le_bytes(bytes);
                    result.push(half::f16::from_bits(f16_bits).to_f32());
                }
                result
            }
            DataType::Q4_0 => {
                // Dequantize Q4_0
                self.dequantize_q4_0(&raw_data, metadata.desc.element_count())?
            }
            DataType::Q8_0 => {
                // Dequantize Q8_0
                self.dequantize_q8_0(&raw_data, metadata.desc.element_count())?
            }
            DataType::Q6_K => {
                // Dequantize Q6_K
                self.dequantize_q6_k(&raw_data, metadata.desc.element_count())?
            }
            _ => {
                return Err(Error::UnsupportedDataType(format!(
                    "Unsupported dtype for loading: {:?}",
                    metadata.desc.dtype
                )))
            }
        };

        // Cache and return
        self.cache.insert(name.to_string(), f32_data);
        Ok(&self.cache[name])
    }

    /// Load multiple tensors at once
    pub fn load_tensors<R: Read + Seek>(
        &mut self,
        names: &[&str],
        parser: &mut GGUFParser<R>,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut result = HashMap::new();
        for &name in names {
            let data = self.load_tensor(name, parser)?.to_vec();
            result.insert(name.to_string(), data);
        }
        Ok(result)
    }

    /// Check if a tensor is loaded in cache
    pub fn is_cached(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }

    /// Get tensor metadata
    pub fn get_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.metadata.get(name)
    }

    /// List all available tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.metadata.keys().map(|s| s.as_str()).collect()
    }

    /// Clear cache to free memory
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size in bytes (approximate)
    pub fn cache_size_bytes(&self) -> usize {
        self.cache.values().map(|v| v.len() * std::mem::size_of::<f32>()).sum()
    }

    // Helper methods for dequantization

    fn dequantize_q4_0(&self, data: &[u8], element_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_0>();
        let mut result = vec![0.0f32; element_count];

        for (block_idx, block_bytes) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            let block: BlockQ4_0 = unsafe { std::ptr::read(block_bytes.as_ptr() as *const _) };

            let offset = block_idx * 32;
            let result_len = result.len();
            let end = (offset + 32).min(result_len);
            if offset < result_len {
                dequantize_q4_0(&block, &mut result[offset..end])?;
            }
        }

        Ok(result)
    }

    fn dequantize_q8_0(&self, data: &[u8], element_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_0>();
        let mut result = vec![0.0f32; element_count];

        for (block_idx, block_bytes) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            let block: BlockQ8_0 = unsafe { std::ptr::read(block_bytes.as_ptr() as *const _) };

            let offset = block_idx * 32;
            let result_len = result.len();
            let end = (offset + 32).min(result_len);
            if offset < result_len {
                dequantize_q8_0(&block, &mut result[offset..end])?;
            }
        }

        Ok(result)
    }

    fn dequantize_q6_k(&self, data: &[u8], element_count: usize) -> Result<Vec<f32>> {
        const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6_K>();
        const QK_K: usize = 256;
        let mut result = vec![0.0f32; element_count];

        for (block_idx, block_bytes) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            let block: BlockQ6_K = unsafe { std::ptr::read(block_bytes.as_ptr() as *const _) };

            let offset = block_idx * QK_K;
            let result_len = result.len();
            let end = (offset + QK_K).min(result_len);
            if offset < result_len {
                dequantize_q6_k(&block, &mut result[offset..end])?;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_loader_creation() {
        let loader = TensorLoader::new(1024);
        assert_eq!(loader.cache_size_bytes(), 0);
        assert_eq!(loader.tensor_names().len(), 0);
    }

    #[test]
    fn test_register_tensor() {
        let mut loader = TensorLoader::new(0);
        let desc = TensorDesc::new(
            "test".to_string(),
            DataType::F32,
            crate::tensor::Shape::new(vec![2, 2]),
            0,
        )
        .unwrap();

        loader.register_tensor("test".to_string(), desc, 0);

        assert_eq!(loader.tensor_names().len(), 1);
        assert!(loader.get_metadata("test").is_some());
    }

    #[test]
    fn test_cache_management() {
        let mut loader = TensorLoader::new(0);

        // Insert some dummy data
        loader.cache.insert("tensor1".to_string(), vec![1.0, 2.0, 3.0]);
        loader.cache.insert("tensor2".to_string(), vec![4.0, 5.0]);

        assert!(loader.is_cached("tensor1"));
        assert!(loader.is_cached("tensor2"));
        assert!(!loader.is_cached("tensor3"));

        let cache_size = loader.cache_size_bytes();
        assert_eq!(cache_size, 5 * std::mem::size_of::<f32>());

        loader.clear_cache();
        assert_eq!(loader.cache_size_bytes(), 0);
    }
}
