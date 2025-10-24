# Quantization Formats Research: Q5_K, Q6_K, Q8_K

**Date:** October 21, 2025  
**Purpose:** Understand Q5_K/Q6_K/Q8_K formats to implement fused kernels with SIMD

---

## 📊 Format Comparison

| Format | Block Size | Bits/Value | Block Size (bytes) | Scales | Complexity |
|--------|-----------|------------|-------------------|---------|------------|
| Q4_K   | 256       | 4          | 144               | Hierarchical (2-level) | ⭐⭐⭐ |
| Q5_K   | 256       | 5          | 176               | Flat (16 scales) | ⭐⭐ |
| Q6_K   | 256       | 6          | 210               | Flat (16 scales) | ⭐⭐⭐ |
| Q8_K   | 256       | 8          | 322               | Flat (32 scales) | ⭐ |

---

## 🔍 Q8_K Format (Simplest)

### Structure
```rust
pub struct BlockQ8_K {
    pub quants: [i8; 256],      // 256 bytes: 8-bit signed values
    pub scales: [u8; 32],       // 32 bytes: 4-bit scales (2 per byte)
    pub d: u16,                 // 2 bytes: f16 super-block scale
    pub dmin: u16,              // 2 bytes: f16 super-block min
}
// Total: 322 bytes
```

### Dequantization Algorithm
```
For 32 groups of 8 values each:
  1. Extract 4-bit scale from scales array:
     - scales[group/2] contains 2 scales (4 bits each)
     - scale = (group % 2 == 0) ? (byte & 0xF) : (byte >> 4)
  
  2. Dequantize 8 values:
     output[i] = d * scale * quants[i] + min
```

**Key Points:**
- **Simplest format:** Direct 8-bit values with flat scales
- **32 groups of 8:** Each group has one 4-bit scale
- **No bit unpacking:** Values are already 8-bit signed integers
- **SIMD-friendly:** Can process 8 values at once easily

---

## 🔍 Q5_K Format (5-bit)

### Structure
```rust
pub struct BlockQ5_K {
    pub ql: [u8; 128],          // 128 bytes: lower 4 bits (2 values per byte)
    pub qh: [u8; 32],           // 32 bytes: upper 1 bit (8 values per byte)
    pub scales: [i8; 16],       // 16 bytes: one scale per 16 values
    pub d: u16,                 // 2 bytes: f16 super-block scale
}
// Total: 178 bytes
```

###

 Dequantization Algorithm
```
For 16 groups of 16 values each:
  1. Reconstruct 5-bit value:
     - 4 bits from ql[idx]
     - 1 bit from qh[idx/8] at position (idx%8)
     - value = ql | (qh_bit << 4)
  
  2. Apply scale:
     output[i] = d * scales[group] * value
```

**Key Points:**
- **5-bit quantization:** 4 bits (lower) + 1 bit (upper)
- **16 groups of 16:** Each group has one 8-bit scale
- **Bit unpacking:** Need to extract and combine bits
- **SIMD challenge:** Bit manipulation can be vectorized

---

## 🔍 Q6_K Format (6-bit)

### Structure
```rust
pub struct BlockQ6_K {
    pub ql: [u8; 128],          // 128 bytes: lower 4 bits (2 values per byte)
    pub qh: [u8; 64],           // 64 bytes: upper 2 bits (4 values per byte)
    pub scales: [i8; 16],       // 16 bytes: one scale per 16 values
    pub d: u16,                 // 2 bytes: f16 super-block scale
}
// Total: 210 bytes
```

### Dequantization Algorithm
```
For 2 halves of 128 values each:
  First half (0-127):
    For l in 0..32:
      q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4) - 32
      q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32
      q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
      q4 = (ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
      
      output[l] = d * scales[is] * q1
      output[l+32] = d * scales[is+2] * q2
      output[l+64] = d * scales[is+4] * q3
      output[l+96] = d * scales[is+6] * q4
  
  Second half (128-255): similar with offset
```

**Key Points:**
- **6-bit quantization:** 4 bits (lower) + 2 bits (upper)
- **16 groups with interleaved layout:** Complex indexing pattern
- **4 values per iteration:** Extracts 4 x 6-bit values at once
- **SIMD challenge:** Complex bit manipulation and interleaving

---

## 🎯 Implementation Strategy

### Order of Implementation
1. **Q8_K first** (1 day) - Simplest, validate approach
2. **Q5_K next** (1 day) - Moderate complexity, 1-bit unpacking
3. **Q6_K last** (1 day) - Most complex, 2-bit unpacking + interleaving

### Common SIMD Patterns

#### Pattern 1: Scalar Multiplication & FMA
```rust
// AVX2: 8x f32 parallel
vacc = _mm256_fmadd_ps(vscale, vquants, vacc);

// NEON: 4x f32 parallel
vacc = vfmaq_f32(vacc, vscale, vquants);
```

#### Pattern 2: Scale Broadcasting
```rust
// AVX2
let vscale = _mm256_set1_ps(scale);

// NEON
let vscale = vdupq_n_f32(scale);
```

#### Pattern 3: Horizontal Sum
```rust
// AVX2
let sum = horizontal_sum_avx2(vacc);

// NEON
let sum = vaddvq_f32(vacc);
```

---

## 🔬 Optimization Opportunities

### Q8_K (Best SIMD Potential)
- ✅ Direct 8-bit values (no bit unpacking)
- ✅ Can load 8 x i8 → convert to 8 x f32
- ✅ Process 8 values per SIMD iteration
- Expected speedup: **2-3x** with SIMD

### Q5_K (Good SIMD Potential)
- ⚠️ Needs bit unpacking (4 + 1 bits)
- ✅ Can extract multiple bits in parallel
- ✅ Process 4-8 values per SIMD iteration
- Expected speedup: **1.5-2x** with SIMD

### Q6_K (Moderate SIMD Potential)
- ⚠️ Complex bit unpacking (4 + 2 bits)
- ⚠️ Interleaved layout complicates vectorization
- ✅ Can still vectorize the FMA operations
- Expected speedup: **1.3-1.7x** with SIMD

---

## 📝 Implementation Checklist

For each format (Q8_K, Q5_K, Q6_K):

### Core Kernel
- [ ] Fused dequant+matmul function
- [ ] Proper hierarchical/flat dequantization
- [ ] Input validation
- [ ] Error handling

### SIMD Optimizations
- [ ] AVX2 implementation (x86-64)
- [ ] NEON implementation (ARM)
- [ ] Scalar fallback
- [ ] Runtime feature detection

### Tests
- [ ] Basic functionality test
- [ ] Correctness test (vs reference)
- [ ] Batch processing test
- [ ] Validation test (error cases)

### Integration
- [ ] Export from `wasm-chord-cpu`
- [ ] Wire up in runtime
- [ ] End-to-end verification

---

## 🎯 Expected Performance Impact

### Current State
- Q4_K: Fused + SIMD ✅
- Q5_K: Naive dequant + matmul ❌
- Q6_K: Naive dequant + matmul ❌
- Q8_K: Naive dequant + matmul ❌

### After Implementation
| Format | Current | Fused+SIMD | Speedup |
|--------|---------|------------|---------|
| Q4_K   | ✅      | ✅         | Baseline |
| Q5_K   | Naive   | ✅         | 2-3x |
| Q6_K   | Naive   | ✅         | 2-3x |
| Q8_K   | Naive   | ✅         | 3-4x |

**Overall Impact:**
- Models using Q5_K/Q6_K/Q8_K will see **2-4x speedup**
- Broader model format support (more GGUF models work well)
- Consistent performance across quantization formats

---

## 🚀 Next Steps

1. **Implement Q8_K fused kernel** (start here - simplest)
2. **Add SIMD for Q8_K** (validate SIMD patterns)
3. **Implement Q5_K fused kernel** (moderate complexity)
4. **Add SIMD for Q5_K** (bit unpacking + SIMD)
5. **Implement Q6_K fused kernel** (most complex)
6. **Add SIMD for Q6_K** (interleaved bit unpacking)
7. **Comprehensive testing** (all formats)
8. **Integration & verification** (end-to-end)

---

**Ready to implement!** Starting with Q8_K (simplest format).

