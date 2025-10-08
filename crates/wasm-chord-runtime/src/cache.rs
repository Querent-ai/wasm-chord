//! Model caching layer for IndexedDB (browser) and filesystem (native)

use wasm_chord_core::error::{Error, Result};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{IdbDatabase, IdbOpenDbRequest, IdbRequest};

#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

/// Model cache key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub model_name: String,
    pub model_hash: String,
}

impl CacheKey {
    pub fn new(model_name: impl Into<String>, model_hash: impl Into<String>) -> Self {
        Self { model_name: model_name.into(), model_hash: model_hash.into() }
    }

    pub fn as_string(&self) -> String {
        format!("{}_{}", self.model_name, self.model_hash)
    }
}

/// Cache storage backend
pub trait CacheBackend {
    /// Store model data in cache
    fn store(&mut self, key: &CacheKey, data: &[u8]) -> Result<()>;

    /// Load model data from cache
    fn load(&self, key: &CacheKey) -> Result<Option<Vec<u8>>>;

    /// Check if model is cached
    fn contains(&self, key: &CacheKey) -> bool;

    /// Remove model from cache
    fn remove(&mut self, key: &CacheKey) -> Result<()>;

    /// Clear entire cache
    fn clear(&mut self) -> Result<()>;

    /// Get cache size in bytes
    fn size(&self) -> Result<u64>;
}

/// Filesystem-based cache (for native/Node.js)
#[cfg(not(target_arch = "wasm32"))]
pub struct FileSystemCache {
    cache_dir: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileSystemCache {
    /// Create a new filesystem cache
    pub fn new(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir).map_err(Error::Io)?;
        Ok(Self { cache_dir })
    }

    /// Get cache file path for key
    fn cache_path(&self, key: &CacheKey) -> PathBuf {
        self.cache_dir.join(format!("{}.cache", key.as_string()))
    }

    /// Get default cache directory
    pub fn default_cache_dir() -> Result<PathBuf> {
        let cache_dir = if let Ok(dir) = std::env::var("WASM_CHORD_CACHE") {
            PathBuf::from(dir)
        } else if let Some(home) = dirs::home_dir() {
            home.join(".cache").join("wasm-chord")
        } else {
            PathBuf::from(".cache/wasm-chord")
        };
        Ok(cache_dir)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl CacheBackend for FileSystemCache {
    fn store(&mut self, key: &CacheKey, data: &[u8]) -> Result<()> {
        let path = self.cache_path(key);
        fs::write(&path, data).map_err(Error::Io)?;
        Ok(())
    }

    fn load(&self, key: &CacheKey) -> Result<Option<Vec<u8>>> {
        let path = self.cache_path(key);
        if path.exists() {
            let data = fs::read(&path).map_err(Error::Io)?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    fn contains(&self, key: &CacheKey) -> bool {
        self.cache_path(key).exists()
    }

    fn remove(&mut self, key: &CacheKey) -> Result<()> {
        let path = self.cache_path(key);
        if path.exists() {
            fs::remove_file(&path).map_err(Error::Io)?;
        }
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        for entry in fs::read_dir(&self.cache_dir).map_err(Error::Io)? {
            let entry = entry.map_err(Error::Io)?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("cache") {
                fs::remove_file(&path).map_err(Error::Io)?;
            }
        }
        Ok(())
    }

    fn size(&self) -> Result<u64> {
        let mut total_size = 0u64;
        for entry in fs::read_dir(&self.cache_dir).map_err(Error::Io)? {
            let entry = entry.map_err(Error::Io)?;
            let metadata = entry.metadata().map_err(Error::Io)?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("cache") {
                total_size += metadata.len();
            }
        }
        Ok(total_size)
    }
}

/// IndexedDB-based cache (for browser)
#[cfg(target_arch = "wasm32")]
pub struct IndexedDBCache {
    db_name: String,
    store_name: String,
}

#[cfg(target_arch = "wasm32")]
impl IndexedDBCache {
    /// Create a new IndexedDB cache
    pub fn new(db_name: impl Into<String>, store_name: impl Into<String>) -> Self {
        Self { db_name: db_name.into(), store_name: store_name.into() }
    }

    /// Initialize the IndexedDB database
    pub async fn init(&self) -> Result<()> {
        // This is a simplified implementation
        // A real implementation would:
        // 1. Open IndexedDB connection
        // 2. Create object store if it doesn't exist
        // 3. Handle version upgrades
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
impl CacheBackend for IndexedDBCache {
    fn store(&mut self, key: &CacheKey, data: &[u8]) -> Result<()> {
        // This is a synchronous stub - real implementation would use async
        // IndexedDB operations are inherently async, so this needs proper
        // wasm-bindgen-futures integration
        Ok(())
    }

    fn load(&self, key: &CacheKey) -> Result<Option<Vec<u8>>> {
        // This is a synchronous stub - real implementation would use async
        Ok(None)
    }

    fn contains(&self, key: &CacheKey) -> bool {
        // This is a synchronous stub - real implementation would use async
        false
    }

    fn remove(&mut self, key: &CacheKey) -> Result<()> {
        // This is a synchronous stub - real implementation would use async
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        // This is a synchronous stub - real implementation would use async
        Ok(())
    }

    fn size(&self) -> Result<u64> {
        // This is a synchronous stub - real implementation would use async
        Ok(0)
    }
}

/// Model cache manager
pub struct ModelCache<B: CacheBackend> {
    backend: B,
}

impl<B: CacheBackend> ModelCache<B> {
    /// Create a new model cache with the given backend
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Store model in cache
    pub fn store(&mut self, key: &CacheKey, data: &[u8]) -> Result<()> {
        self.backend.store(key, data)
    }

    /// Load model from cache
    pub fn load(&self, key: &CacheKey) -> Result<Option<Vec<u8>>> {
        self.backend.load(key)
    }

    /// Check if model is cached
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.backend.contains(key)
    }

    /// Remove model from cache
    pub fn remove(&mut self, key: &CacheKey) -> Result<()> {
        self.backend.remove(key)
    }

    /// Clear entire cache
    pub fn clear(&mut self) -> Result<()> {
        self.backend.clear()
    }

    /// Get cache size in bytes
    pub fn size(&self) -> Result<u64> {
        self.backend.size()
    }

    /// Load model with caching
    /// If cached, loads from cache. Otherwise, loads from loader and caches.
    pub fn load_with_cache<F>(&mut self, key: &CacheKey, loader: F) -> Result<Vec<u8>>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        if let Some(cached_data) = self.load(key)? {
            return Ok(cached_data);
        }

        let data = loader()?;
        self.store(key, &data)?;
        Ok(data)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ModelCache<FileSystemCache> {
    /// Create a new filesystem-based model cache with default directory
    pub fn with_default_backend() -> Result<Self> {
        let cache_dir = FileSystemCache::default_cache_dir()?;
        let backend = FileSystemCache::new(cache_dir)?;
        Ok(Self::new(backend))
    }
}

#[cfg(target_arch = "wasm32")]
impl ModelCache<IndexedDBCache> {
    /// Create a new IndexedDB-based model cache with default configuration
    pub fn with_default_backend() -> Self {
        let backend = IndexedDBCache::new("wasm-chord-cache", "models");
        Self::new(backend)
    }
}

// Helper for dirs crate (only needed for non-wasm)
#[cfg(not(target_arch = "wasm32"))]
mod dirs {
    use std::path::PathBuf;

    pub fn home_dir() -> Option<PathBuf> {
        std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_cache_key() {
        let key = CacheKey::new("tinyllama", "abc123");
        assert_eq!(key.as_string(), "tinyllama_abc123");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_filesystem_cache() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("wasm-chord-test-cache");
        let mut cache = FileSystemCache::new(&temp_dir)?;

        let key = CacheKey::new("test-model", "hash123");
        let data = b"test model data";

        // Store
        cache.store(&key, data)?;
        assert!(cache.contains(&key));

        // Load
        let loaded = cache.load(&key)?;
        assert_eq!(loaded, Some(data.to_vec()));

        // Size
        let size = cache.size()?;
        assert!(size >= data.len() as u64);

        // Remove
        cache.remove(&key)?;
        assert!(!cache.contains(&key));

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }
}
