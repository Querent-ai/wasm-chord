/// Model caching test
/// Tests filesystem-based model caching
use std::path::PathBuf;
use wasm_chord_runtime::{CacheKey, FileSystemCache, ModelCache};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗄️  Model Cache Test");
    println!("===================\n");

    // Create a temporary cache directory
    let cache_dir = PathBuf::from("/tmp/wasm-chord-cache-test");
    println!("📂 Cache directory: {:?}", cache_dir);

    // Clean up any previous test data
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir)?;
    }

    // Create cache backend
    let backend = FileSystemCache::new(&cache_dir)?;
    let mut cache = ModelCache::new(backend);
    println!("✅ Cache initialized\n");

    // Test 1: Store and load model
    println!("📝 Test 1: Store and load model");
    let key1 = CacheKey::new("tinyllama-1.1b", "abc123");
    let test_data = b"This is test model data for TinyLlama";

    cache.store(&key1, test_data)?;
    println!("   ✓ Stored {} bytes", test_data.len());

    assert!(cache.contains(&key1));
    println!("   ✓ Cache contains key");

    let loaded = cache.load(&key1)?;
    assert_eq!(loaded, Some(test_data.to_vec()));
    println!("   ✓ Loaded data matches\n");

    // Test 2: Multiple models
    println!("📝 Test 2: Multiple models");
    let key2 = CacheKey::new("llama2-7b", "def456");
    let test_data2 = b"This is test model data for Llama2";

    cache.store(&key2, test_data2)?;
    println!("   ✓ Stored second model");

    assert!(cache.contains(&key1));
    assert!(cache.contains(&key2));
    println!("   ✓ Both models cached\n");

    // Test 3: Cache size
    println!("📝 Test 3: Cache size");
    let size = cache.size()?;
    println!("   Cache size: {} bytes", size);
    assert!(size >= (test_data.len() + test_data2.len()) as u64);
    println!("   ✓ Size is correct\n");

    // Test 4: Load with cache (miss)
    println!("📝 Test 4: Load with cache (miss)");
    let key3 = CacheKey::new("new-model", "xyz789");
    let loader_called = std::cell::Cell::new(false);

    let loaded_data = cache.load_with_cache(&key3, || {
        loader_called.set(true);
        Ok(b"New model data from loader".to_vec())
    })?;

    assert!(loader_called.get());
    println!("   ✓ Loader was called (cache miss)");
    assert_eq!(loaded_data, b"New model data from loader");
    println!("   ✓ Data from loader is correct\n");

    // Test 5: Load with cache (hit)
    println!("📝 Test 5: Load with cache (hit)");
    let loader_called2 = std::cell::Cell::new(false);

    let loaded_data2 = cache.load_with_cache(&key3, || {
        loader_called2.set(true);
        Ok(b"This should not be called".to_vec())
    })?;

    assert!(!loader_called2.get());
    println!("   ✓ Loader was NOT called (cache hit)");
    assert_eq!(loaded_data2, b"New model data from loader");
    println!("   ✓ Data from cache is correct\n");

    // Test 6: Remove model
    println!("📝 Test 6: Remove model");
    cache.remove(&key1)?;
    assert!(!cache.contains(&key1));
    println!("   ✓ Model removed from cache\n");

    // Test 7: Clear cache
    println!("📝 Test 7: Clear cache");
    cache.clear()?;
    assert!(!cache.contains(&key2));
    assert!(!cache.contains(&key3));
    println!("   ✓ All models cleared\n");

    // Test 8: Default cache directory
    println!("📝 Test 8: Default cache directory");
    let _default_cache = ModelCache::<FileSystemCache>::with_default_backend()?;
    let default_dir = FileSystemCache::default_cache_dir()?;
    println!("   Default cache directory: {:?}", default_dir);
    println!("   ✓ Default cache created successfully\n");

    // Cleanup
    println!("🧹 Cleaning up test directory...");
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir)?;
    }

    println!("\n✅ All cache tests passed!");
    println!("🎉 Model caching is working correctly!");

    Ok(())
}
