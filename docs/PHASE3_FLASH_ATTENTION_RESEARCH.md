# ⚡ Flash Attention Research Summary

**Date:** 2025-10-21
**Purpose:** Research findings for implementing Flash Attention in wasm-chord
**Status:** Research Phase Complete

---

## 🎯 What is Flash Attention?

Flash Attention is an **IO-aware** attention algorithm that achieves:
- **3-4x faster** attention computation
- **10x less memory** usage (O(N) instead of O(N²))
- **Exact results** (not an approximation)
- **Longer sequences** (32K+ tokens vs 2K limit)

### Key Innovation: IO-Awareness

**Problem with Standard Attention:**
```
Standard Attention Flow (SLOW):
1. Load Q, K, V from HBM → SRAM
2. Compute QK^T → Write to HBM (N² memory!)
3. Load QK^T from HBM → Compute Softmax → Write to HBM
4. Load Softmax from HBM → Load V → Compute Output → Write to HBM

❌ Multiple HBM read/writes (SLOW - 100x slower than SRAM)
❌ Stores N² attention matrix in HBM (HUGE memory)
```

**Flash Attention Flow (FAST):**
```
Flash Attention (IO-Aware):
1. Load Q, K, V blocks into SRAM (small blocks)
2. Compute attention IN SRAM (fused operations)
3. Write only final output to HBM

✅ Minimal HBM access (only input/output)
✅ Never materialize full N² attention matrix
✅ All computation in fast SRAM
```

---

## 📊 Algorithm Details

### Core Technique: Tiling + Online Softmax

**Tiling:**
- Break Q, K, V into small blocks (e.g., 128×128)
- Process one block at a time in SRAM
- Accumulate results incrementally

**Online Softmax:**
- Compute softmax incrementally without storing full matrix
- Track running max and sum for numerical stability
- Update output as each block is processed

### Mathematical Foundation

**Standard Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d) V

Memory: O(N²) for QK^T matrix
HBM Access: 4N² reads + 4N² writes
```

**Flash Attention:**
```
Same formula, but computed in blocks:
1. Divide Q into blocks: Q₁, Q₂, ..., Q_Tc
2. Divide K, V into blocks: K₁, K₂, ..., K_Tr
3. For each Q block:
   - For each K, V block:
     - Load Q_i, K_j, V_j into SRAM
     - Compute S_ij = Q_i K_j^T / √d
     - Update running softmax statistics
     - Accumulate output: O_i += softmax(S_ij) V_j
4. Final output is exact attention

Memory: O(N) for incremental statistics
HBM Access: O(N²/M) where M is SRAM size
```

---

## 🔧 Implementation Requirements

### 1. Memory Hierarchy Awareness

**GPU Memory Hierarchy:**
```
HBM (High Bandwidth Memory):
- Size: 8-80 GB
- Speed: 1-2 TB/s
- Latency: ~100 ns

SRAM (On-chip Cache):
- Size: 20-40 MB per SM
- Speed: 19-30 TB/s (10-20x faster!)
- Latency: ~1 ns

Goal: Keep data in SRAM as much as possible
```

### 2. Block Size Selection

**Optimal Block Size:**
- Must fit in SRAM: Q_block + K_block + V_block + Output ≤ SRAM size
- Typical: 64×64 to 256×256
- Trade-off: Larger = fewer HBM accesses, but must fit in SRAM

**For our implementation:**
```rust
// NVIDIA A100: 192 KB SRAM per SM
// Block size: 128 × 128 (configurable)
// Memory per block:
//   Q: 128 × 128 × 2 bytes (fp16) = 32 KB
//   K: 128 × 128 × 2 bytes = 32 KB
//   V: 128 × 128 × 2 bytes = 32 KB
//   Temp: ~50 KB
//   Total: ~150 KB ✅ Fits in SRAM
```

### 3. Kernel Fusion

**Operations to Fuse:**
1. QK^T computation
2. Scaling by 1/√d
3. Softmax (online version)
4. Multiplication by V
5. Output accumulation

**All must happen in one kernel, without writing intermediates to HBM**

---

## 💻 Implementation Architecture

### File Structure
```
crates/wasm-chord-runtime/src/
├── attention/
│   ├── mod.rs              # Attention trait
│   ├── standard.rs         # Current (standard attention)
│   ├── flash.rs           # NEW: Flash Attention
│   └── config.rs          # Attention config (block size, etc.)
├── gpu/
│   ├── cuda/
│   │   ├── flash_attn.cu          # CUDA kernel
│   │   └── flash_attn_forward.cuh # Forward pass
│   ├── metal/
│   │   └── flash_attn.metal       # Metal shader
│   └── webgpu/
│       └── flash_attn.wgsl        # WebGPU compute
└── transformer/
    └── attention.rs        # Modified to use Flash Attention
```

### Rust API Design

```rust
pub trait Attention {
    fn forward(
        &self,
        q: &Tensor,      // [batch, seq_len, num_heads, head_dim]
        k: &Tensor,      // [batch, seq_len_kv, num_heads, head_dim]
        v: &Tensor,      // [batch, seq_len_kv, num_heads, head_dim]
        mask: Option<&Tensor>,
    ) -> Result<Tensor>; // [batch, seq_len, num_heads, head_dim]
}

pub struct FlashAttention {
    block_size_q: usize,     // Default: 128
    block_size_kv: usize,    // Default: 128
    num_splits: usize,       // For parallelization
    backend: AttentionBackend,
}

pub enum AttentionBackend {
    Cuda(CudaFlashAttn),
    Metal(MetalFlashAttn),
    WebGPU(WebGPUFlashAttn),
    CPU(StandardAttention),  // Fallback
}
```

---

## 🚀 CUDA Implementation Pseudocode

```cuda
__global__ void flash_attention_forward(
    const half* Q,     // [batch, num_heads, seq_len_q, head_dim]
    const half* K,     // [batch, num_heads, seq_len_k, head_dim]
    const half* V,     // [batch, num_heads, seq_len_k, head_dim]
    half* O,           // [batch, num_heads, seq_len_q, head_dim]
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    int block_size
) {
    // Shared memory for blocks
    __shared__ half Q_block[BLOCK_SIZE][HEAD_DIM];
    __shared__ half K_block[BLOCK_SIZE][HEAD_DIM];
    __shared__ half V_block[BLOCK_SIZE][HEAD_DIM];
    __shared__ half S_block[BLOCK_SIZE][BLOCK_SIZE];

    // Online softmax statistics
    float m_prev = -INFINITY;  // running max
    float l_prev = 0.0f;        // running sum

    // Output accumulator
    float O_local[HEAD_DIM] = {0};

    // Loop over K, V blocks
    for (int j = 0; j < num_kv_blocks; j++) {
        // Load K, V blocks into shared memory
        load_block(K, K_block, j * block_size);
        load_block(V, V_block, j * block_size);
        __syncthreads();

        // Load Q block
        load_block(Q, Q_block, blockIdx.x * block_size);
        __syncthreads();

        // Compute S = Q @ K^T / sqrt(d)
        matmul_shared(Q_block, K_block, S_block, scale);
        __syncthreads();

        // Online softmax update
        float m_curr = -INFINITY;
        float l_curr = 0.0f;

        // Find max in this block
        for (int i = 0; i < block_size; i++) {
            m_curr = fmaxf(m_curr, S_block[threadIdx.x][i]);
        }

        // Compute exp and sum
        for (int i = 0; i < block_size; i++) {
            float exp_val = expf(S_block[threadIdx.x][i] - m_curr);
            S_block[threadIdx.x][i] = exp_val;
            l_curr += exp_val;
        }

        // Update global statistics
        float m_new = fmaxf(m_prev, m_curr);
        float l_new = expf(m_prev - m_new) * l_prev +
                      expf(m_curr - m_new) * l_curr;

        // Rescale previous output
        float scale_o = expf(m_prev - m_new) * l_prev / l_new;
        for (int d = 0; d < head_dim; d++) {
            O_local[d] *= scale_o;
        }

        // Add contribution from current block
        float scale_v = expf(m_curr - m_new) / l_new;
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < block_size; i++) {
                sum += S_block[threadIdx.x][i] * V_block[i][d];
            }
            O_local[d] += scale_v * sum;
        }

        // Update statistics for next iteration
        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // Write final output
    write_output(O, O_local, blockIdx.x * block_size + threadIdx.x);
}
```

---

## 📈 Expected Performance

### Comparison with Standard Attention

| Sequence Length | Standard | Flash Attn | Speedup |
|----------------|----------|------------|---------|
| **512** | 10ms | 3ms | 3.3x |
| **1024** | 40ms | 12ms | 3.3x |
| **2048** | 160ms | 45ms | 3.6x |
| **4096** | 640ms | 170ms | 3.8x |
| **8192** | OOM | 650ms | ∞ |

### Memory Usage

| Sequence Length | Standard | Flash Attn | Reduction |
|----------------|----------|------------|-----------|
| **1024** | 4 MB | 0.4 MB | 10x |
| **2048** | 16 MB | 0.8 MB | 20x |
| **4096** | 64 MB | 1.6 MB | 40x |
| **8192** | 256 MB | 3.2 MB | 80x |

---

## 🎯 Implementation Roadmap

### Day 1-2: Core Algorithm (IN PROGRESS)
- [x] Research Flash Attention paper ✅
- [x] Understand tiling and online softmax ✅
- [ ] Design Rust API
- [ ] Implement block-wise attention logic
- [ ] Add online softmax computation

### Day 3-4: CUDA Kernel
- [ ] Write CUDA kernel skeleton
- [ ] Implement shared memory management
- [ ] Add online softmax in CUDA
- [ ] Optimize memory access patterns
- [ ] Add FP16 support

### Day 5: Metal Shader
- [ ] Port algorithm to Metal
- [ ] Optimize for Apple Silicon
- [ ] Add threadgroup memory usage
- [ ] Test on M1/M2/M3

### Day 6: Testing & Benchmarking
- [ ] Unit tests (correctness vs standard)
- [ ] Numerical stability tests
- [ ] Performance benchmarks
- [ ] Memory usage verification
- [ ] Tune block sizes

---

## 📚 Reference Implementations

### To Study:
1. **Original Flash Attention** (PyTorch + CUDA)
   - Repo: https://github.com/Dao-AILab/flash-attention
   - Key file: `csrc/flash_attn/flash_api.cpp`

2. **Triton Implementation** (Educational)
   - Tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
   - Good for understanding algorithm

3. **xFormers** (Production-ready)
   - Repo: https://github.com/facebookresearch/xformers
   - Well-optimized kernels

4. **vLLM Flash Attention**
   - Repo: https://github.com/vllm-project/vllm
   - Production inference optimizations

---

## 🔍 Key Insights

### What Makes Flash Attention Fast?

1. **IO-Awareness**: Minimizes slow HBM accesses
2. **Kernel Fusion**: All ops in one kernel (no intermediate writes)
3. **Tiling**: Processes small blocks that fit in fast SRAM
4. **Online Softmax**: No need to store full attention matrix
5. **Optimal Complexity**: O(N²/M) HBM accesses vs O(N²) standard

### Implementation Challenges

1. **Memory Management**: Must carefully manage SRAM usage
2. **Numerical Stability**: Online softmax needs careful handling
3. **Block Size Tuning**: Optimal size varies by hardware
4. **Multi-GPU**: Need to shard across GPUs efficiently

### Why It's Worth It

- **Enables Longer Sequences**: 32K+ tokens vs 2K limit
- **3-4x Faster**: Directly translates to faster inference
- **Lower Cost**: Less memory = smaller/cheaper GPUs
- **Production Ready**: Used by GPT-4, Claude, Llama, etc.

---

## 🚀 Next Steps

**Immediate (Today):**
1. Create `attention/flash.rs` with API design
2. Implement block-wise tiling logic in Rust
3. Add online softmax computation
4. Write CPU reference implementation

**Tomorrow:**
1. Start CUDA kernel implementation
2. Add shared memory management
3. Implement fused operations
4. Test correctness vs reference

**This Week:**
1. Complete CUDA kernel
2. Add Metal shader
3. Benchmark performance
4. Tune block sizes
5. Integrate with transformer

---

## 📖 Mathematical Details

### Online Softmax Algorithm

```
Given: S_ij = score for Q_i and K_j block

Initialize:
  m = -∞   (running max)
  l = 0     (running sum)
  O = 0     (output accumulator)

For each K, V block j:
  1. Compute S_ij = Q_i @ K_j^T / √d

  2. Find new max:
     m_new = max(m, max(S_ij))

  3. Rescale previous output:
     O ← O × exp(m - m_new) × (l / l_new)

  4. Compute new sum:
     l_new = exp(m - m_new) × l + Σ exp(S_ij - m_new)

  5. Add contribution from this block:
     O ← O + (softmax(S_ij, m_new) @ V_j)

  6. Update statistics:
     m ← m_new
     l ← l_new

Final: O is the exact attention output
```

### Why This Works

- **Incrementally updates softmax** without storing full matrix
- **Numerically stable** by tracking running max
- **Exact results** (not an approximation)
- **O(1) memory** per output position (vs O(N) for standard)

---

## 🎉 Summary

Flash Attention is a **game-changer** for transformer inference:

✅ **3-4x faster** through IO-aware design
✅ **10x less memory** with online softmax
✅ **Exact results** (not approximate)
✅ **Production proven** (used in GPT-4, Claude, etc.)
✅ **Well understood** (multiple reference implementations)

**We're ready to implement this in wasm-chord!** 🚀

Next: Start coding `attention/flash.rs`
