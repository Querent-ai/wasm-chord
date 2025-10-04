/// GGUF (GPT-Generated Unified Format) streaming parser
///
/// Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
use crate::error::{Error, Result};
use crate::tensor::{DataType, Shape, TensorDesc};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom};

/// GGUF magic number (version 3)
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;

/// Model metadata extracted from GGUF header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub architecture: String,
    pub vocab_size: Option<u32>,
    pub tensors: Vec<TensorDesc>,
}

/// GGUF streaming parser
pub struct GGUFParser<R: Read + Seek> {
    reader: R,
    meta: Option<ModelMeta>,
}

impl<R: Read + Seek> GGUFParser<R> {
    pub fn new(reader: R) -> Self {
        Self { reader, meta: None }
    }

    /// Parse GGUF header and extract metadata
    pub fn parse_header(&mut self) -> Result<ModelMeta> {
        // Read magic
        let magic = self.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::InvalidFormat(format!("Invalid GGUF magic: 0x{:08X}", magic)));
        }

        // Read version
        let version = self.read_u32()?;
        if version != GGUF_VERSION {
            return Err(Error::InvalidFormat(format!("Unsupported GGUF version: {}", version)));
        }

        // Read counts
        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;

        // Parse metadata key-values (simplified - skip for now)
        // In a real implementation, we'd parse these to extract architecture, vocab, etc.
        let metadata = self.parse_metadata(metadata_kv_count)?;

        // Parse tensor info
        let tensors = self.parse_tensor_info(tensor_count)?;

        let meta = ModelMeta {
            version,
            tensor_count,
            metadata_kv_count,
            architecture: metadata
                .get("general.architecture")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
            vocab_size: metadata.get("vocab_size").and_then(|v| v.parse().ok()),
            tensors,
        };

        self.meta = Some(meta.clone());
        Ok(meta)
    }

    /// Get current position for tensor data
    pub fn tensor_data_offset(&mut self) -> Result<u64> {
        self.reader.stream_position().map_err(Error::from)
    }

    /// Read tensor data chunk
    pub fn read_tensor_data(&mut self, offset: u64, size: usize) -> Result<Vec<u8>> {
        self.reader.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; size];
        self.reader.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Get data offset (for TensorLoader)
    pub fn data_offset(&self) -> u64 {
        // This would be calculated after parsing header
        // For now, return a placeholder
        0
    }

    /// Get tensor information
    pub fn tensor_info(&self) -> Vec<(String, TensorDesc, u64)> {
        if let Some(meta) = &self.meta {
            meta.tensors
                .iter()
                .enumerate()
                .map(|(i, desc)| (format!("tensor_{}", i), desc.clone(), 0))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get metadata (if parsed)
    pub fn metadata(&self) -> Option<&ModelMeta> {
        self.meta.as_ref()
    }

    // Helper methods

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| Error::ParseError(format!("Invalid UTF-8: {}", e)))
    }

    fn parse_metadata(&mut self, count: u64) -> Result<std::collections::HashMap<String, String>> {
        let mut map = std::collections::HashMap::new();

        for _ in 0..count {
            let key = self.read_string()?;

            // Read value type (simplified - assume string for now)
            let value_type = self.read_u32()?;

            // For simplicity, skip actual value parsing
            // In production, we'd properly parse based on value_type
            let _ = value_type;

            // Placeholder
            map.insert(key, "placeholder".to_string());
        }

        Ok(map)
    }

    fn parse_tensor_info(&mut self, count: u64) -> Result<Vec<TensorDesc>> {
        let mut tensors = Vec::new();
        let mut current_offset = 0u64;

        for _ in 0..count {
            let name = self.read_string()?;

            // Read n_dims
            let n_dims = self.read_u32()? as usize;

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(self.read_u64()? as usize);
            }

            // Read dtype
            let dtype_val = self.read_u32()?;
            let dtype = self.parse_dtype(dtype_val)?;

            // Read offset (relative to tensor data section)
            let offset = self.read_u64()?;

            let shape = Shape::new(dims);
            let desc = TensorDesc::new(name, dtype, shape, current_offset + offset)?;

            current_offset += desc.size_bytes as u64;
            tensors.push(desc);
        }

        Ok(tensors)
    }

    fn parse_dtype(&self, val: u32) -> Result<DataType> {
        match val {
            0 => Ok(DataType::F32),
            1 => Ok(DataType::F16),
            2 => Ok(DataType::Q4_0),
            3 => Ok(DataType::Q4_1),
            8 => Ok(DataType::Q8_0),
            9 => Ok(DataType::Q8_1),
            _ => Err(Error::UnsupportedDataType(format!("Unknown dtype: {}", val))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_gguf_magic() {
        // Create minimal GGUF header
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        let cursor = Cursor::new(data);
        let mut parser = GGUFParser::new(cursor);

        let meta = parser.parse_header().unwrap();
        assert_eq!(meta.version, GGUF_VERSION);
        assert_eq!(meta.tensor_count, 0);
    }
}
