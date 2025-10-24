//! Production-grade memory pool allocator for tensor buffers
//!
//! This module provides a high-performance memory pool that reduces malloc/free
//! overhead and improves cache locality for tensor operations.

use std::alloc::{Allocator, Global, Layout};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::thread_local;

/// Thread-local memory pool for high-performance tensor allocation
pub struct MemoryPool {
    /// Pool of pre-allocated buffers by size
    pools: RefCell<HashMap<usize, Vec<PooledBuffer>>>,
    /// Statistics for monitoring
    stats: RefCell<PoolStats>,
}

/// A pooled buffer with metadata
#[derive(Debug)]
struct PooledBuffer {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
    in_use: bool,
}

/// Memory pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub total_bytes_allocated: usize,
    pub total_bytes_pooled: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self {
            pools: RefCell::new(HashMap::new()),
            stats: RefCell::new(PoolStats::default()),
        }
    }

    /// Allocate a buffer of the specified size
    pub fn allocate(&self, size: usize) -> Option<PooledBuffer> {
        let mut pools = self.pools.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        // Try to find a buffer in the pool
        if let Some(buffer_list) = pools.get_mut(&size) {
            if let Some(buffer) = buffer_list.pop() {
                stats.pool_hits += 1;
                stats.allocations += 1;
                stats.total_bytes_allocated += size;
                return Some(buffer);
            }
        }

        // Pool miss - allocate new buffer
        stats.pool_misses += 1;
        stats.allocations += 1;
        stats.total_bytes_allocated += size;
        stats.total_bytes_pooled += size;

        let layout = Layout::from_size_align(size, 8).ok()?;
        let ptr = unsafe { Global.allocate(layout).ok()?.as_ptr() };
        
        Some(PooledBuffer {
            ptr: NonNull::new(ptr)?,
            size,
            layout,
            in_use: true,
        })
    }

    /// Deallocate a buffer back to the pool
    pub fn deallocate(&self, mut buffer: PooledBuffer) {
        let mut pools = self.pools.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        buffer.in_use = false;
        stats.deallocations += 1;

        // Add back to pool (limit pool size to prevent memory bloat)
        let max_pool_size = 10;
        if let Some(buffer_list) = pools.get_mut(&buffer.size) {
            if buffer_list.len() < max_pool_size {
                buffer_list.push(buffer);
                return;
            }
        } else {
            pools.insert(buffer.size, vec![buffer]);
            return;
        }

        // Pool is full, actually deallocate
        unsafe {
            Global.deallocate(buffer.ptr, buffer.layout);
        }
        stats.total_bytes_pooled -= buffer.size;
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.borrow().clone()
    }

    /// Clear all pooled buffers (for testing/memory pressure)
    pub fn clear(&self) {
        let mut pools = self.pools.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        for (_, buffer_list) in pools.iter_mut() {
            for buffer in buffer_list.drain(..) {
                unsafe {
                    Global.deallocate(buffer.ptr, buffer.layout);
                }
                stats.total_bytes_pooled -= buffer.size;
            }
        }
    }
}

/// Thread-local memory pool instance
thread_local! {
    static MEMORY_POOL: MemoryPool = MemoryPool::new();
}

/// Allocate a tensor buffer using the memory pool
pub fn allocate_tensor_buffer(size: usize) -> Option<NonNull<u8>> {
    MEMORY_POOL.with(|pool| {
        pool.allocate(size).map(|buffer| buffer.ptr)
    })
}

/// Deallocate a tensor buffer back to the pool
pub fn deallocate_tensor_buffer(ptr: NonNull<u8>, size: usize) {
    MEMORY_POOL.with(|pool| {
        let layout = Layout::from_size_align(size, 8).unwrap();
        let buffer = PooledBuffer {
            ptr,
            size,
            layout,
            in_use: false,
        };
        pool.deallocate(buffer);
    });
}

/// Get memory pool statistics
pub fn get_pool_stats() -> PoolStats {
    MEMORY_POOL.with(|pool| pool.stats())
}

/// Clear the memory pool (for testing)
pub fn clear_memory_pool() {
    MEMORY_POOL.with(|pool| pool.clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        clear_memory_pool();
        
        // Allocate buffer
        let ptr1 = allocate_tensor_buffer(1024).unwrap();
        assert!(!ptr1.as_ptr().is_null());
        
        // Deallocate
        deallocate_tensor_buffer(ptr1, 1024);
        
        // Allocate same size again - should reuse from pool
        let ptr2 = allocate_tensor_buffer(1024).unwrap();
        assert_eq!(ptr1, ptr2);
        
        let stats = get_pool_stats();
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_memory_pool_different_sizes() {
        clear_memory_pool();
        
        let ptr1 = allocate_tensor_buffer(512).unwrap();
        let ptr2 = allocate_tensor_buffer(1024).unwrap();
        
        assert_ne!(ptr1, ptr2);
        
        deallocate_tensor_buffer(ptr1, 512);
        deallocate_tensor_buffer(ptr2, 1024);
        
        let stats = get_pool_stats();
        assert_eq!(stats.pool_misses, 2);
    }

    #[test]
    fn test_memory_pool_stats() {
        clear_memory_pool();
        
        let ptr = allocate_tensor_buffer(2048).unwrap();
        deallocate_tensor_buffer(ptr, 2048);
        
        let stats = get_pool_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.deallocations, 1);
        assert_eq!(stats.total_bytes_allocated, 2048);
    }
}
