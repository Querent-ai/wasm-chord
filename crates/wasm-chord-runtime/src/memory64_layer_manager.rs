//! Memory64 Layer Loading System
//!
//! This module implements on-demand layer loading from Memory64 storage,
//! enabling inference with models larger than 4GB by loading layers
//! into WASM memory only when needed.

use crate::memory64::Memory64Runtime;
use crate::transformer::{TransformerConfig, TransformerLayer};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wasm_chord_core::error::{Error, Result}; // For LRU eviction

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    pub cached_layers: usize,
    pub max_cache_size: usize,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub evictions: u32,
    pub prefetch_protected_evictions: u32,
}

/// Lazy tensor loader trait for on-demand weight loading
pub trait LazyTensorLoader {
    fn load_tensor(&mut self, tensor_name: &str) -> Result<Vec<f32>>;
}

/// Memory64-aware layer manager with lazy loading
pub struct Memory64LayerManager {
    /// Memory64 runtime for accessing stored layers
    #[allow(dead_code)]
    runtime: Arc<Memory64Runtime>,
    /// Model configuration
    config: TransformerConfig,
    /// Cache for loaded layers (layer_id -> (layer_data, last_accessed_time))
    layer_cache: HashMap<u32, (TransformerLayer, Instant)>,
    /// Maximum number of layers to keep in cache
    max_cache_size: usize,
    /// Cache statistics
    stats: CacheStats,
    /// Lazy tensor loader for on-demand weight loading
    tensor_loader: Option<Box<dyn LazyTensorLoader>>,
    /// Layers protected from eviction due to prefetching
    prefetch_protected: std::collections::HashSet<u32>,
    /// Current prefetch distance (for protection calculation)
    prefetch_distance: u32,
}

impl Memory64LayerManager {
    /// Create a new Memory64 layer manager
    pub fn new(
        runtime: Arc<Memory64Runtime>,
        config: TransformerConfig,
        max_cache_size: usize,
    ) -> Self {
        Self {
            runtime,
            config,
            layer_cache: HashMap::new(),
            max_cache_size,
            stats: CacheStats { max_cache_size, ..Default::default() },
            tensor_loader: None,
            prefetch_protected: std::collections::HashSet::new(),
            prefetch_distance: 1,
        }
    }

    /// Set the lazy tensor loader
    pub fn set_tensor_loader(&mut self, loader: Box<dyn LazyTensorLoader>) {
        self.tensor_loader = Some(loader);
    }

    /// Load a specific layer from Memory64 into WASM memory
    pub fn load_layer(&mut self, layer_id: u32) -> Result<&TransformerLayer> {
        // Check if layer is already cached
        if self.layer_cache.contains_key(&layer_id) {
            self.stats.cache_hits += 1;
            let (layer, last_accessed) = self.layer_cache.get_mut(&layer_id).unwrap();
            *last_accessed = Instant::now(); // Update last accessed time
            return Ok(layer);
        }

        self.stats.cache_misses += 1;
        // Load layer from Memory64
        println!("🔄 Loading layer {} from Memory64...", layer_id);

        // In production, this would call the Memory64 runtime to load the layer
        // For now, we'll create a placeholder layer
        let layer_data = self.load_layer_data_from_memory64(layer_id)?;

        // Parse layer data into TransformerLayer
        let layer = self.parse_layer_data(layer_id, &layer_data)?;

        // Cache management - remove oldest layers if cache is full
        if self.layer_cache.len() >= self.max_cache_size {
            self.smart_evict_layer();
        }

        // Store in cache with timestamp
        self.layer_cache.insert(layer_id, (layer, Instant::now()));
        self.stats.cached_layers = self.layer_cache.len();

        // Return reference to cached layer
        Ok(&self.layer_cache.get(&layer_id).unwrap().0)
    }

    /// Load layer data from Memory64 storage
    fn load_layer_data_from_memory64(&self, layer_id: u32) -> Result<Vec<f32>> {
        // In production, this would use the Memory64 runtime to read the layer data
        // For now, we'll simulate loading with placeholder data

        let config = &self.config;
        let hidden_size = config.hidden_size;

        // Calculate expected data size for this layer
        let attention_size = hidden_size * hidden_size * 4; // 4 attention matrices
        let ffn_size = hidden_size * (config.hidden_size * 4) * 2; // 2 FFN matrices
        let norm_size = hidden_size * 2; // 2 normalization layers

        let total_size = attention_size + ffn_size + norm_size;

        // Create placeholder data (in production, this would be real layer data)
        let mut data = vec![0.0; total_size];

        // Fill with some recognizable pattern for testing
        for (i, val) in data.iter_mut().enumerate() {
            *val = (layer_id as f32) * 0.1 + (i as f32) * 0.001;
        }

        println!("   📦 Loaded {} bytes for layer {}", total_size, layer_id);
        Ok(data)
    }

    /// Parse raw layer data into TransformerLayer structure
    fn parse_layer_data(&self, layer_id: u32, data: &[f32]) -> Result<TransformerLayer> {
        // This is a simplified parser - in production, you'd parse the actual GGUF format
        // For now, we'll create a placeholder layer with the correct dimensions

        let config = &self.config;
        let hidden_size = config.hidden_size;

        // Calculate expected data size for this layer
        let attention_size = hidden_size * hidden_size * 4; // 4 attention matrices
        let ffn_size = hidden_size * (config.hidden_size * 4) * 2; // 2 FFN matrices
        let norm_size = hidden_size * 2; // 2 normalization layers

        let expected_size = attention_size + ffn_size + norm_size;

        if data.len() < expected_size {
            return Err(Error::ParseError(format!(
                "Layer {} data too small: expected {}, got {}",
                layer_id,
                expected_size,
                data.len()
            )));
        }

        // Create layer with placeholder data
        // In production, you'd parse the actual tensor data from GGUF format
        let mut layer = TransformerLayer::new(config);

        // Fill with actual data (simplified - in production, parse GGUF tensors)
        let mut offset = 0;

        // Attention weights (simplified)
        if offset + attention_size <= data.len() {
            // In production: parse attention_q, attention_k, attention_v, attention_o
            offset += attention_size;
        }

        // FFN weights (simplified)
        if offset + ffn_size <= data.len() {
            // In production: parse ffn_gate, ffn_up, ffn_down
            offset += ffn_size;
        }

        // Normalization layers
        if offset + norm_size <= data.len() {
            // attention_norm
            layer.attention_norm.copy_from_slice(&data[offset..offset + hidden_size]);
            offset += hidden_size;

            // ffn_norm
            layer.ffn_norm.copy_from_slice(&data[offset..offset + hidden_size]);
        }

        println!("✅ Layer {} loaded successfully ({} bytes)", layer_id, data.len());
        Ok(layer)
    }

    /// Smart eviction that considers prefetch protection
    fn smart_evict_layer(&mut self) {
        if self.layer_cache.is_empty() {
            return;
        }

        // Find the best candidate for eviction
        let eviction_candidate = self.find_eviction_candidate();

        if let Some(layer_id) = eviction_candidate {
            self.layer_cache.remove(&layer_id);
            self.stats.evictions += 1;
            self.stats.cached_layers = self.layer_cache.len();
            println!("🗑️  Evicted layer {} from cache", layer_id);
        }
    }

    /// Find the best layer to evict, considering prefetch protection
    fn find_eviction_candidate(&self) -> Option<u32> {
        // First, try to find unprotected layers
        let unprotected_candidates: Vec<u32> = self
            .layer_cache
            .keys()
            .filter(|&&layer_id| !self.prefetch_protected.contains(&layer_id))
            .copied()
            .collect();

        if !unprotected_candidates.is_empty() {
            // Find the oldest unprotected layer
            return unprotected_candidates
                .iter()
                .min_by_key(|&&layer_id| {
                    self.layer_cache.get(&layer_id).map(|(_, time)| time).unwrap()
                })
                .copied();
        }

        // If all layers are protected, evict the oldest protected layer
        // This prevents cache from becoming completely stuck
        self.layer_cache
            .iter()
            .min_by_key(|(_, (_, last_accessed))| last_accessed)
            .map(|(layer_id, _)| *layer_id)
    }

    /// Evict the oldest layer from cache (fallback method)
    fn evict_oldest_layer(&mut self) {
        if let Some(oldest_key) = self.layer_cache.keys().next().copied() {
            self.layer_cache.remove(&oldest_key);
            self.stats.evictions += 1;
            self.stats.cached_layers = self.layer_cache.len();
            println!("🗑️  Evicted layer {} from cache", oldest_key);
        }
    }

    /// Mark layers as protected from eviction due to prefetching
    pub fn protect_prefetch_layers(&mut self, current_layer: u32) {
        self.prefetch_protected.clear();

        if self.prefetch_distance == 0 {
            return;
        }

        // Protect the next N layers that are likely to be accessed
        let max_layer = (current_layer + self.prefetch_distance)
            .min(self.config.num_layers.saturating_sub(1) as u32);
        for layer_id in (current_layer + 1)..=max_layer {
            self.prefetch_protected.insert(layer_id);
        }

        println!(
            "🛡️  Protected {} layers from eviction (prefetch distance: {})",
            self.prefetch_protected.len(),
            self.prefetch_distance
        );
    }

    /// Set prefetch distance for protection calculation
    pub fn set_prefetch_distance(&mut self, distance: u32) {
        self.prefetch_distance = distance;
    }

    /// Get current prefetch distance
    pub fn get_prefetch_distance(&self) -> u32 {
        self.prefetch_distance
    }

    /// Clear prefetch protection (call when prefetch is complete)
    pub fn clear_prefetch_protection(&mut self) {
        self.prefetch_protected.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear the layer cache
    pub fn clear_cache(&mut self) {
        self.layer_cache.clear();
        self.stats.cached_layers = 0;
        println!("🧹 Layer cache cleared");
    }

    /// Set the maximum cache size
    pub fn set_cache_size(&mut self, new_size: usize) {
        let old_size = self.max_cache_size;
        self.max_cache_size = new_size;
        self.stats.max_cache_size = new_size;

        println!("📊 Cache size changed: {} -> {} layers", old_size, new_size);

        // Evict excess layers if new size is smaller
        if self.layer_cache.len() > new_size {
            let excess = self.layer_cache.len() - new_size;
            for _ in 0..excess {
                self.evict_oldest_layer();
            }
            println!("🗑️ Evicted {} excess layers", excess);
        }
    }

    /// Get the current cache size
    pub fn get_cache_size(&self) -> usize {
        self.max_cache_size
    }

    /// Get the current number of cached layers
    pub fn get_cached_layers_count(&self) -> usize {
        self.layer_cache.len()
    }

    /// Calculate optimal cache size based on available memory
    pub fn calculate_optimal_cache_size(&self, available_memory_mb: u64) -> usize {
        // Estimate memory per layer (roughly 50MB for TinyLlama-like models)
        let memory_per_layer_mb = 50;
        let max_layers_by_memory = (available_memory_mb / memory_per_layer_mb) as usize;

        // Cap at reasonable limits
        let optimal_size = max_layers_by_memory.clamp(4, 16);

        println!("🧮 Optimal cache size calculation:");
        println!("   Available memory: {} MB", available_memory_mb);
        println!("   Memory per layer: {} MB", memory_per_layer_mb);
        println!("   Calculated optimal: {} layers", optimal_size);

        optimal_size
    }

    /// Auto-configure cache size based on system memory
    pub fn auto_configure_cache_size(&mut self) {
        // Try to detect available system memory
        let available_memory_mb = self.detect_available_memory();
        let optimal_size = self.calculate_optimal_cache_size(available_memory_mb);
        self.set_cache_size(optimal_size);
    }

    /// Detect available system memory (simplified)
    fn detect_available_memory(&self) -> u64 {
        // This is a simplified implementation
        // In production, you'd use system APIs to detect available memory

        // Default to 1GB for safety
        let default_memory_mb = 1024;

        // Try to detect from environment or system
        if let Ok(memory_str) = std::env::var("WASM_CHORD_CACHE_MEMORY_MB") {
            if let Ok(memory_mb) = memory_str.parse::<u64>() {
                return memory_mb;
            }
        }

        default_memory_mb
    }

    /// Preload multiple layers (useful for batch processing)
    pub fn preload_layers(&mut self, layer_ids: &[u32]) -> Result<()> {
        for &layer_id in layer_ids {
            self.load_layer(layer_id)?;
        }
        Ok(())
    }
}

/// Memory64-aware model that loads layers on-demand
pub struct Memory64Model {
    /// Model configuration
    pub config: TransformerConfig,
    /// Token embeddings (always loaded)
    pub token_embeddings: Vec<f32>,
    /// Output normalization (always loaded)
    pub output_norm: Vec<f32>,
    /// LM head (always loaded)
    pub lm_head: Vec<f32>,
    /// Layer manager for on-demand loading
    layer_manager: Memory64LayerManager,
    /// Number of layers in the model
    num_layers: u32,
    /// Prefetch distance: how many subsequent layers to pre-load after access
    prefetch_distance: u32,
}

impl Memory64Model {
    /// Create a new Memory64-aware model
    pub fn new(
        config: TransformerConfig,
        token_embeddings: Vec<f32>,
        output_norm: Vec<f32>,
        lm_head: Vec<f32>,
        layer_manager: Memory64LayerManager,
        num_layers: u32,
    ) -> Self {
        Self {
            config,
            token_embeddings,
            output_norm,
            lm_head,
            layer_manager,
            num_layers,
            prefetch_distance: 1,
        }
    }

    /// Get a specific layer (loads on-demand)
    pub fn get_layer(&mut self, layer_id: u32) -> Result<&TransformerLayer> {
        if layer_id >= self.num_layers {
            return Err(Error::ParseError(format!(
                "Layer {} out of range (max: {})",
                layer_id,
                self.num_layers - 1
            )));
        }

        // Protect prefetch layers from eviction
        self.layer_manager.protect_prefetch_layers(layer_id);

        // Ensure requested layer is loaded
        let _ = self.layer_manager.load_layer(layer_id)?;

        // Simple synchronous prefetch of subsequent layers (best-effort).
        // Important: prefetch before returning a reference to avoid borrow conflicts.
        if self.prefetch_distance > 0 {
            let max_next =
                (layer_id + self.prefetch_distance).min(self.num_layers.saturating_sub(1));
            for next_id in (layer_id + 1)..=max_next {
                let _ = self.layer_manager.load_layer(next_id);
            }
        }

        // Clear prefetch protection after prefetch is complete
        self.layer_manager.clear_prefetch_protection();

        // Return reference after prefetching is complete
        self.layer_manager.load_layer(layer_id)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        self.layer_manager.cache_stats()
    }

    /// Clear layer cache
    pub fn clear_cache(&mut self) {
        self.layer_manager.clear_cache();
    }

    /// Preload all layers (useful for batch processing)
    pub fn preload_all_layers(&mut self) -> Result<()> {
        let layer_ids: Vec<u32> = (0..self.num_layers).collect();
        self.layer_manager.preload_layers(&layer_ids)
    }

    /// Get the number of layers in the model
    pub fn num_layers(&self) -> u32 {
        self.num_layers
    }

    /// Configure prefetch distance (0 disables prefetch)
    pub fn set_prefetch_distance(&mut self, distance: u32) {
        self.prefetch_distance = distance;
    }

    /// Get the current prefetch distance
    pub fn get_prefetch_distance(&self) -> u32 {
        self.prefetch_distance
    }

    /// Configure cache size
    pub fn set_cache_size(&mut self, size: usize) {
        self.layer_manager.set_cache_size(size);
    }

    /// Get current cache size
    pub fn get_cache_size(&self) -> usize {
        self.layer_manager.get_cache_size()
    }

    /// Auto-configure cache size based on available memory
    pub fn auto_configure_cache_size(&mut self) {
        self.layer_manager.auto_configure_cache_size();
    }

    /// Get cache utilization statistics
    pub fn get_cache_utilization(&self) -> (usize, usize) {
        let cached = self.layer_manager.get_cached_layers_count();
        let max = self.layer_manager.get_cache_size();
        (cached, max)
    }
}
