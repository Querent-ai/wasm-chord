//! Multi-threading improvements for tensor operations
//!
//! This module provides work-stealing and parallel layer processing
//! for maximum CPU utilization.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Work-stealing scheduler for attention blocks
pub struct WorkStealingScheduler {
    /// Number of worker threads
    num_threads: usize,
    /// Work queue for stealing
    work_queue: Arc<std::sync::Mutex<Vec<WorkItem>>>,
    /// Statistics
    stats: Arc<WorkStats>,
}

/// A unit of work for the scheduler
#[derive(Debug, Clone)]
pub struct WorkItem {
    pub batch_idx: usize,
    pub out_idx: usize,
    pub block_idx: usize,
    pub group_idx: usize,
    pub priority: u32, // Higher = more important
}

/// Work-stealing statistics
#[derive(Debug, Default)]
pub struct WorkStats {
    pub work_items_processed: AtomicUsize,
    pub work_items_stolen: AtomicUsize,
    pub total_work_time_ms: AtomicUsize,
    pub thread_utilization: AtomicUsize,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub fn new() -> Self {
        let num_threads = num_cpus::get();
        Self {
            num_threads,
            work_queue: Arc::new(std::sync::Mutex::new(Vec::new())),
            stats: Arc::new(WorkStats::default()),
        }
    }

    /// Schedule work items for parallel processing
    pub fn schedule_work(&self, work_items: Vec<WorkItem>) {
        let mut queue = self.work_queue.lock().unwrap();
        queue.extend(work_items);
        queue.sort_by(|a, b| b.priority.cmp(&a.priority)); // Higher priority first
    }

    /// Process work items in parallel with work-stealing
    pub fn process_work<F>(&self, mut work_fn: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(WorkItem) -> Result<(), Box<dyn std::error::Error>> + Send + Sync,
    {
        let work_fn = Arc::new(std::sync::Mutex::new(work_fn));
        
        // Process work items in parallel
        let result: Result<Vec<_>, _> = (0..self.num_threads)
            .into_par_iter()
            .map(|thread_id| {
                let mut local_work = Vec::new();
                
                // Steal work from the global queue
                while let Some(work_item) = self.steal_work() {
                    local_work.push(work_item);
                }
                
                // Process local work
                for work_item in local_work {
                    let start = std::time::Instant::now();
                    
                    // Execute work
                    let mut work_fn = work_fn.lock().unwrap();
                    work_fn(work_item)?;
                    
                    let duration = start.elapsed();
                    self.stats.total_work_time_ms.fetch_add(
                        duration.as_millis() as usize,
                        Ordering::Relaxed,
                    );
                    self.stats.work_items_processed.fetch_add(1, Ordering::Relaxed);
                }
                
                Ok(())
            })
            .collect();
        
        result?;
        Ok(())
    }

    /// Steal work from the global queue
    fn steal_work(&self) -> Option<WorkItem> {
        let mut queue = self.work_queue.lock().unwrap();
        queue.pop()
    }

    /// Get work statistics
    pub fn stats(&self) -> &WorkStats {
        &self.stats
    }
}

/// Parallel layer processor for transformer layers
pub struct ParallelLayerProcessor {
    /// Number of worker threads
    num_threads: usize,
    /// Statistics
    stats: Arc<LayerStats>,
}

/// Layer processing statistics
#[derive(Debug, Default)]
pub struct LayerStats {
    pub layers_processed: AtomicUsize,
    pub total_layer_time_ms: AtomicUsize,
    pub parallel_efficiency: AtomicUsize,
}

impl ParallelLayerProcessor {
    /// Create a new parallel layer processor
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus::get(),
            stats: Arc::new(LayerStats::default()),
        }
    }

    /// Process multiple layers in parallel
    pub fn process_layers<F>(
        &self,
        layer_indices: Vec<usize>,
        mut layer_fn: F,
    ) -> Result<Vec<()>, Box<dyn std::error::Error>>
    where
        F: FnMut(usize) -> Result<(), Box<dyn std::error::Error>> + Send + Sync,
    {
        let layer_fn = Arc::new(std::sync::Mutex::new(layer_fn));
        
        let results: Result<Vec<_>, _> = layer_indices
            .into_par_iter()
            .map(|layer_idx| {
                let start = std::time::Instant::now();
                
                // Process layer
                let mut layer_fn = layer_fn.lock().unwrap();
                let result = layer_fn(layer_idx);
                
                let duration = start.elapsed();
                self.stats.total_layer_time_ms.fetch_add(
                    duration.as_millis() as usize,
                    Ordering::Relaxed,
                );
                self.stats.layers_processed.fetch_add(1, Ordering::Relaxed);
                
                result
            })
            .collect();
        
        results
    }

    /// Process attention blocks in parallel within a layer
    pub fn process_attention_blocks<F>(
        &self,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        mut block_fn: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize, usize, usize) -> Result<(), Box<dyn std::error::Error>> + Send + Sync,
    {
        let block_fn = Arc::new(std::sync::Mutex::new(block_fn));
        
        // Create work items for all attention blocks
        let work_items: Vec<_> = (0..batch_size)
            .flat_map(|batch_idx| {
                (0..num_heads)
                    .flat_map(move |head_idx| {
                        (0..seq_len)
                            .map(move |seq_idx| {
                                // Calculate priority based on position
                                let priority = (batch_idx * num_heads * seq_len + head_idx * seq_len + seq_idx) as u32;
                                WorkItem {
                                    batch_idx,
                                    out_idx: head_idx,
                                    block_idx: seq_idx,
                                    group_idx: 0,
                                    priority,
                                }
                            })
                    })
            })
            .collect();
        
        // Process work items in parallel
        work_items
            .into_par_iter()
            .map(|work_item| {
                let mut block_fn = block_fn.lock().unwrap();
                block_fn(
                    work_item.batch_idx,
                    work_item.out_idx,
                    work_item.block_idx,
                    work_item.group_idx,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(())
    }

    /// Get layer processing statistics
    pub fn stats(&self) -> &LayerStats {
        &self.stats
    }
}

/// Optimized parallel matrix multiplication
pub struct ParallelMatMul {
    /// Number of worker threads
    num_threads: usize,
    /// Statistics
    stats: Arc<MatMulStats>,
}

/// Matrix multiplication statistics
#[derive(Debug, Default)]
pub struct MatMulStats {
    pub matmuls_processed: AtomicUsize,
    pub total_matmul_time_ms: AtomicUsize,
    pub parallel_efficiency: AtomicUsize,
}

impl ParallelMatMul {
    /// Create a new parallel matrix multiplication processor
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus::get(),
            stats: Arc::new(MatMulStats::default()),
        }
    }

    /// Perform parallel matrix multiplication
    pub fn matmul_parallel<F>(
        &self,
        batch_size: usize,
        output_features: usize,
        mut matmul_fn: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize) -> Result<(), Box<dyn std::error::Error>> + Send + Sync,
    {
        let matmul_fn = Arc::new(std::sync::Mutex::new(matmul_fn));
        
        // Process matrix multiplication in parallel
        let work_items: Vec<_> = (0..batch_size)
            .flat_map(|batch_idx| {
                (0..output_features)
                    .map(move |out_idx| (batch_idx, out_idx))
            })
            .collect();
        
        work_items
            .into_par_iter()
            .map(|(batch_idx, out_idx)| {
                let start = std::time::Instant::now();
                
                let mut matmul_fn = matmul_fn.lock().unwrap();
                let result = matmul_fn(batch_idx, out_idx);
                
                let duration = start.elapsed();
                self.stats.total_matmul_time_ms.fetch_add(
                    duration.as_millis() as usize,
                    Ordering::Relaxed,
                );
                self.stats.matmuls_processed.fetch_add(1, Ordering::Relaxed);
                
                result
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(())
    }

    /// Get matrix multiplication statistics
    pub fn stats(&self) -> &MatMulStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_stealing_scheduler() {
        let scheduler = WorkStealingScheduler::new();
        assert!(scheduler.num_threads > 0);
    }

    #[test]
    fn test_parallel_layer_processor() {
        let processor = ParallelLayerProcessor::new();
        assert!(processor.num_threads > 0);
    }

    #[test]
    fn test_parallel_matmul() {
        let matmul = ParallelMatMul::new();
        assert!(matmul.num_threads > 0);
    }
}
