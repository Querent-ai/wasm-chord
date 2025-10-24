//! Cache-aware blocking optimizations for tensor operations
//!
//! This module provides cache-optimized blocking strategies for Q4_K/Q6_K
//! block processing to maximize L1/L2/L3 cache utilization.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache configuration for different CPU architectures
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 data cache size in bytes
    pub l1_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L3 cache size in bytes
    pub l3_size: usize,
    /// Cache line size in bytes
    pub cache_line_size: usize,
    /// Optimal block size for Q4_K processing
    pub q4k_block_size: usize,
    /// Optimal block size for Q6_K processing
    pub q6k_block_size: usize,
}

impl CacheConfig {
    /// Detect cache configuration at runtime
    pub fn detect() -> Self {
        // Default values for modern CPUs
        let mut config = CacheConfig {
            l1_size: 32 * 1024,      // 32KB L1D
            l2_size: 256 * 1024,     // 256KB L2
            l3_size: 8 * 1024 * 1024, // 8MB L3
            cache_line_size: 64,     // 64-byte cache lines
            q4k_block_size: 64,      // Will be calculated
            q6k_block_size: 64,      // Will be calculated
        };

        // Try to detect actual cache sizes (simplified detection)
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // Modern Intel/AMD CPUs
                config.l1_size = 32 * 1024;
                config.l2_size = 256 * 1024;
                config.l3_size = 8 * 1024 * 1024;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                // Modern ARM CPUs (Apple M1/M2, etc.)
                config.l1_size = 64 * 1024;   // Larger L1 on ARM
                config.l2_size = 512 * 1024;  // Larger L2 on ARM
                config.l3_size = 16 * 1024 * 1024; // Larger L3 on ARM
            }
        }

        // Calculate optimal block sizes
        config.q4k_block_size = Self::calculate_q4k_block_size(&config);
        config.q6k_block_size = Self::calculate_q6k_block_size(&config);

        config
    }

    /// Calculate optimal Q4_K block size based on cache configuration
    fn calculate_q4k_block_size(config: &CacheConfig) -> usize {
        // Q4_K block: 144 bytes per 256 elements
        // We want to fit multiple blocks in L1 cache
        let q4k_block_bytes = 144;
        let l1_blocks = config.l1_size / q4k_block_bytes;
        
        // Use 75% of L1 cache to leave room for input/output data
        let optimal_blocks = (l1_blocks * 3) / 4;
        optimal_blocks.max(4).min(32) // Between 4 and 32 blocks
    }

    /// Calculate optimal Q6_K block size based on cache configuration
    fn calculate_q6k_block_size(config: &CacheConfig) -> usize {
        // Q6_K block: 210 bytes per 256 elements
        let q6k_block_bytes = 210;
        let l1_blocks = config.l1_size / q6k_block_bytes;
        
        // Use 75% of L1 cache to leave room for input/output data
        let optimal_blocks = (l1_blocks * 3) / 4;
        optimal_blocks.max(4).min(24) // Between 4 and 24 blocks
    }
}

/// Cache-aware Q4_K block processor
pub struct CacheAwareQ4KProcessor {
    config: CacheConfig,
    /// Statistics for monitoring cache performance
    stats: CacheStats,
}

/// Cache performance statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    pub l1_hits: AtomicUsize,
    pub l2_hits: AtomicUsize,
    pub l3_hits: AtomicUsize,
    pub memory_accesses: AtomicUsize,
    pub blocks_processed: AtomicUsize,
}

impl CacheAwareQ4KProcessor {
    /// Create a new cache-aware Q4_K processor
    pub fn new() -> Self {
        Self {
            config: CacheConfig::detect(),
            stats: CacheStats::default(),
        }
    }

    /// Process Q4_K blocks with cache-aware blocking
    pub fn process_blocks<F>(
        &self,
        blocks: &[wasm_chord_core::quant::BlockQ4_K],
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        num_output_features: usize,
        k: usize,
        mut process_fn: F,
    ) -> Result<(), wasm_chord_core::error::Error>
    where
        F: FnMut(
            &[wasm_chord_core::quant::BlockQ4_K],
            &[f32],
            &mut [f32],
            usize,
            usize,
            usize,
        ) -> Result<(), wasm_chord_core::error::Error>,
    {
        let num_blocks_per_row = k / wasm_chord_core::quant::QK_K;
        let block_size = self.config.q4k_block_size;

        // Process in cache-friendly blocks
        for batch_idx in 0..batch_size {
            let input_row = &input[batch_idx * k..(batch_idx + 1) * k];
            let output_row = &mut output[batch_idx * num_output_features..(batch_idx + 1) * num_output_features];

            // Process output features in blocks
            for out_block_start in (0..num_output_features).step_by(block_size) {
                let out_block_end = (out_block_start + block_size).min(num_output_features);
                let out_block_size = out_block_end - out_block_start;

                // Process K dimension in blocks
                for k_block_start in (0..num_blocks_per_row).step_by(block_size) {
                    let k_block_end = (k_block_start + block_size).min(num_blocks_per_row);
                    let k_block_size = k_block_end - k_block_start;

                    // Extract relevant blocks for this cache block
                    let mut cache_blocks = Vec::with_capacity(out_block_size * k_block_size);
                    for out_idx in out_block_start..out_block_end {
                        for k_idx in k_block_start..k_block_end {
                            let block_idx = out_idx * num_blocks_per_row + k_idx;
                            cache_blocks.push(blocks[block_idx]);
                        }
                    }

                    // Process this cache block
                    let input_offset = k_block_start * wasm_chord_core::quant::QK_K;
                    let input_block = &input_row[input_offset..input_offset + k_block_size * wasm_chord_core::quant::QK_K];
                    let output_block = &mut output_row[out_block_start..out_block_end];

                    process_fn(
                        &cache_blocks,
                        input_block,
                        output_block,
                        1, // Single batch element for this block
                        out_block_size,
                        k_block_size * wasm_chord_core::quant::QK_K,
                    )?;

                    self.stats.blocks_processed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Get cache performance statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}

/// Cache-aware Q6_K block processor
pub struct CacheAwareQ6KProcessor {
    config: CacheConfig,
    stats: CacheStats,
}

impl CacheAwareQ6KProcessor {
    /// Create a new cache-aware Q6_K processor
    pub fn new() -> Self {
        Self {
            config: CacheConfig::detect(),
            stats: CacheStats::default(),
        }
    }

    /// Process Q6_K blocks with cache-aware blocking
    pub fn process_blocks<F>(
        &self,
        blocks: &[wasm_chord_core::quant::BlockQ6_K],
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        num_output_features: usize,
        k: usize,
        mut process_fn: F,
    ) -> Result<(), wasm_chord_core::error::Error>
    where
        F: FnMut(
            &[wasm_chord_core::quant::BlockQ6_K],
            &[f32],
            &mut [f32],
            usize,
            usize,
            usize,
        ) -> Result<(), wasm_chord_core::error::Error>,
    {
        let num_blocks_per_row = k / wasm_chord_core::quant::QK_K;
        let block_size = self.config.q6k_block_size;

        // Process in cache-friendly blocks
        for batch_idx in 0..batch_size {
            let input_row = &input[batch_idx * k..(batch_idx + 1) * k];
            let output_row = &mut output[batch_idx * num_output_features..(batch_idx + 1) * num_output_features];

            // Process output features in blocks
            for out_block_start in (0..num_output_features).step_by(block_size) {
                let out_block_end = (out_block_start + block_size).min(num_output_features);
                let out_block_size = out_block_end - out_block_start;

                // Process K dimension in blocks
                for k_block_start in (0..num_blocks_per_row).step_by(block_size) {
                    let k_block_end = (k_block_start + block_size).min(num_blocks_per_row);
                    let k_block_size = k_block_end - k_block_start;

                    // Extract relevant blocks for this cache block
                    let mut cache_blocks = Vec::with_capacity(out_block_size * k_block_size);
                    for out_idx in out_block_start..out_block_end {
                        for k_idx in k_block_start..k_block_end {
                            let block_idx = out_idx * num_blocks_per_row + k_idx;
                            cache_blocks.push(blocks[block_idx]);
                        }
                    }

                    // Process this cache block
                    let input_offset = k_block_start * wasm_chord_core::quant::QK_K;
                    let input_block = &input_row[input_offset..input_offset + k_block_size * wasm_chord_core::quant::QK_K];
                    let output_block = &mut output_row[out_block_start..out_block_end];

                    process_fn(
                        &cache_blocks,
                        input_block,
                        output_block,
                        1, // Single batch element for this block
                        out_block_size,
                        k_block_size * wasm_chord_core::quant::QK_K,
                    )?;

                    self.stats.blocks_processed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Get cache performance statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_detection() {
        let config = CacheConfig::detect();
        assert!(config.l1_size > 0);
        assert!(config.l2_size > 0);
        assert!(config.l3_size > 0);
        assert!(config.cache_line_size > 0);
        assert!(config.q4k_block_size > 0);
        assert!(config.q6k_block_size > 0);
    }

    #[test]
    fn test_q4k_processor_creation() {
        let processor = CacheAwareQ4KProcessor::new();
        assert!(processor.config().q4k_block_size > 0);
    }

    #[test]
    fn test_q6k_processor_creation() {
        let processor = CacheAwareQ6KProcessor::new();
        assert!(processor.config().q6k_block_size > 0);
    }
}
