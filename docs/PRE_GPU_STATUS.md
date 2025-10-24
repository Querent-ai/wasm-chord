# Pre-GPU Checklist Status

**Date:** October 23, 2025

---

## ✅ Item #1: End-to-End Benchmark - PARTIAL COMPLETE

**Created:** `examples/end-to-end-benchmark`

### What Works:
- ✅ Model loading (10-22 seconds for TinyLlama 1.1B)
- ✅ Memory measurement (6.3 GB for Q4_K model)
- ✅ Tokenization (prompt encoding works)

### What Doesn't Work:
- ❌ Inference crashes (NaN probabilities in sampling)
- ❌ Can't measure tokens/second yet
- ❌ Most weights fall back to legacy loading ("IO error: failed to fill whole buffer")

### Key Findings:
```
Model: TinyLlama 1.1B Q4_K_M
Load time: ~10-22 seconds
Memory: 6,264 MB
Optimal loading: Only token_embd, output_norm, output.weight succeed
Legacy fallback: All layer weights (blk.*.*)
```

### Issues Discovered:
1. **Optimal loading fails** for most weights with "IO error: failed to fill whole buffer"
2. **NaN values** in probability distributions causing sampling crashes
3. **Negative probabilities** in weighted sampling

### Next Steps for #1:
- Debug why optimal loading fails (tensor offset calculation?)
- Fix NaN probability issue in model forward pass
- Add option to disable optimal loading for testing

---

## 🔄 Item #2: Verify Dispatch Usage - IN PROGRESS

Need to confirm fused kernels are actually being used in production.

### Plan:
1. Add `DEBUG_DISPATCH` environment variable to `dispatch_matmul`
2. Run inference with debug enabled
3. Verify Q4_K/Q5_K/Q6_K use fused paths
4. Check for unexpected F32 fallbacks

---

## ⏳ Item #3: Multi-Model Validation - PENDING

Need 7B model to test (only have TinyLlama 1.1B).

---

## ⏳ Item #4: Memory Leak Testing - PENDING

Can't test until inference works properly.

---

## 📊 Summary

**Completed:** 1/4 (partial)
**In Progress:** 1/4
**Pending:** 2/4

**Blocking Issues:**
1. Optimal weight loading fails for layer weights
2. Model inference produces NaN probabilities

**Ready for GPU:** ❌ Not yet - need to fix CPU baseline first

---

## 🎯 Immediate Actions

1. ✅ **Add dispatch verification** (Item #2)
2. ⏳ **Debug optimal loading failures**
3. ⏳ **Fix NaN probability issue**
4. ⏳ **Get baseline CPU numbers**
