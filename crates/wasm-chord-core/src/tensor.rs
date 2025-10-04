use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F16,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32,
    /// 4-bit quantized (group-wise)
    Q4_0,
    Q4_1,
    /// 8-bit quantized (group-wise)
    Q8_0,
    Q8_1,
}

impl DataType {
    /// Size in bytes per element (for non-quantized types)
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F16 | DataType::I16 | DataType::U16 => 2,
            DataType::I8 | DataType::U8 => 1,
            DataType::Q4_0 | DataType::Q4_1 => 0, // Variable, handled separately
            DataType::Q8_0 | DataType::Q8_1 => 1,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, DataType::Q4_0 | DataType::Q4_1 | DataType::Q8_0 | DataType::Q8_1)
    }
}

/// Tensor shape (dimensions)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Validates shape dimensions
    pub fn validate(&self) -> Result<()> {
        if self.0.is_empty() {
            return Err(Error::InvalidShape("Shape cannot be empty".into()));
        }
        if self.0.contains(&0) {
            return Err(Error::InvalidShape("Shape dimensions must be > 0".into()));
        }
        Ok(())
    }
}

/// Tensor descriptor (metadata about a tensor in memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDesc {
    /// Tensor name (e.g., "model.layers.0.attention.wq")
    pub name: String,
    /// Data type
    pub dtype: DataType,
    /// Shape
    pub shape: Shape,
    /// Offset in memory (bytes)
    pub offset: u64,
    /// Total size in bytes
    pub size_bytes: usize,
    /// Layout (row-major, col-major, packed)
    pub layout: TensorLayout,
    /// Quantization metadata (if applicable)
    pub quant_meta: Option<QuantMeta>,
}

impl TensorDesc {
    pub fn new(name: String, dtype: DataType, shape: Shape, offset: u64) -> Result<Self> {
        shape.validate()?;

        let size_bytes = if dtype.is_quantized() {
            // Placeholder: actual size depends on group size
            shape.numel() / 2 // Approximate for Q4
        } else {
            shape.numel() * dtype.size()
        };

        Ok(Self {
            name,
            dtype,
            shape,
            offset,
            size_bytes,
            layout: TensorLayout::RowMajor,
            quant_meta: None,
        })
    }

    /// Get total number of elements in the tensor
    pub fn element_count(&self) -> usize {
        self.shape.numel()
    }
}

/// Tensor memory layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorLayout {
    RowMajor,
    ColMajor,
    PackedBlock,
}

/// Quantization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantMeta {
    /// Group size for quantization
    pub group_size: usize,
    /// Scale factors (per group)
    pub scales_offset: u64,
    /// Zero points (per group, if applicable)
    pub zeros_offset: Option<u64>,
}

/// Tensor view (runtime representation)
pub struct Tensor {
    pub desc: TensorDesc,
    /// Pointer to data (offset in shared memory)
    pub data_ptr: *const u8,
}

impl Tensor {
    /// Create a tensor view from descriptor and pointer
    pub fn new(desc: TensorDesc, data_ptr: *const u8) -> Self {
        Self { desc, data_ptr }
    }

    /// Get raw data slice
    ///
    /// # Safety
    /// Caller must ensure that data_ptr is valid and points to at least size_bytes of memory.
    pub unsafe fn data_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.data_ptr, self.desc.size_bytes)
    }
}

// Tensor is not Send/Sync by default due to raw pointer
// This is intentional - multi-threading requires explicit synchronization
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_validation() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert!(shape.validate().is_ok());
        assert_eq!(shape.numel(), 24);

        let invalid = Shape::new(vec![2, 0, 4]);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DataType::F32.size(), 4);
        assert_eq!(DataType::F16.size(), 2);
        assert_eq!(DataType::I8.size(), 1);
    }

    #[test]
    fn test_tensor_desc() {
        let desc =
            TensorDesc::new("test.weight".into(), DataType::F32, Shape::new(vec![128, 256]), 0)
                .unwrap();

        assert_eq!(desc.size_bytes, 128 * 256 * 4);
        assert_eq!(desc.shape.numel(), 128 * 256);
    }
}
