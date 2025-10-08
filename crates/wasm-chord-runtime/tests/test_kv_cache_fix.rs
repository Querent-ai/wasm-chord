/// Test to verify KV cache bug fixes
use wasm_chord_runtime::KVCache;

#[test]
fn test_kv_cache_append_tracking() {
    // Create a small cache: max_seq_len=4, num_kv_heads=2, head_dim=4
    let max_seq_len = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let mut cache = KVCache::new(max_seq_len, num_kv_heads, head_dim);

    println!(
        "Initial: current_seq_len={}, max_seq_len={}",
        cache.current_seq_len, cache.max_seq_len
    );
    assert_eq!(cache.current_seq_len, 0);
    assert_eq!(cache.max_seq_len, 4); // 4 tokens max

    // Scenario 1: Prefill with 2 tokens
    // Each token: 2 kv_heads * 4 head_dim = 8 elements
    let keys_prefill = vec![1.0; 16]; // 2 tokens * 8 = 16 elements
    let vals_prefill = vec![2.0; 16];

    let _ = cache.append(&keys_prefill, &vals_prefill);
    println!("After prefill: current_seq_len={}", cache.current_seq_len);
    assert_eq!(cache.current_seq_len, 2, "current_seq_len should be 2 tokens after prefill");

    // Verify data was written correctly
    assert_eq!(cache.keys[0], 1.0);
    assert_eq!(cache.keys[15], 1.0);
    assert_eq!(cache.values[0], 2.0);

    // Scenario 2: Incremental generation (1 token)
    let keys_inc = vec![3.0; 8]; // 1 token * 8 = 8 elements
    let vals_inc = vec![4.0; 8];

    let _ = cache.append(&keys_inc, &vals_inc);
    println!("After incremental: current_seq_len={}", cache.current_seq_len);
    assert_eq!(cache.current_seq_len, 3, "current_seq_len should be 3 tokens after incremental");

    // Verify incremental data was appended, not overwriting
    assert_eq!(cache.keys[0], 1.0, "Prefill data should not be overwritten");
    assert_eq!(cache.keys[15], 1.0, "Prefill data should not be overwritten");
    assert_eq!(cache.keys[16], 3.0, "Incremental data should start at position 16");
    assert_eq!(cache.keys[23], 3.0, "Incremental data should end at position 23");
    assert_eq!(cache.values[16], 4.0);

    println!("✅ KV cache append tracking works correctly!");
}

#[test]
fn test_kv_cache_slicing() {
    // Test that we can correctly slice the cache to get only valid tokens
    let mut cache = KVCache::new(4, 2, 4);

    // Add 2 tokens
    let _ = cache.append(&[1.0; 16], &[2.0; 16]);

    // current_seq_len tracks number of tokens
    assert_eq!(cache.current_seq_len, 2, "Should have 2 tokens in cache");

    // Get valid portion - need to convert token count to element count
    let num_kv_heads = 2;
    let head_dim = 4;
    let valid_element_count = cache.current_seq_len * num_kv_heads * head_dim;
    let valid_keys = &cache.keys[..valid_element_count];
    assert_eq!(valid_keys.len(), 16, "Should have 16 valid elements");

    println!("✅ KV cache slicing works correctly!");
}
