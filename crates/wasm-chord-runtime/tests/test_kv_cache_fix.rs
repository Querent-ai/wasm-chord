/// Test to verify KV cache bug fixes
use wasm_chord_runtime::KVCache;

#[test]
fn test_kv_cache_append_tracking() {
    // Create a small cache: max_seq_len=4, num_kv_heads=2, head_dim=4
    let max_seq_len = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let mut cache = KVCache::new(max_seq_len, num_kv_heads, head_dim);

    println!("Initial: seq_pos={}, max_size={}", cache.seq_pos, cache.max_size);
    assert_eq!(cache.seq_pos, 0);
    assert_eq!(cache.max_size, 4 * 2 * 4); // 32 elements

    // Scenario 1: Prefill with 2 tokens
    // Each token: 2 kv_heads * 4 head_dim = 8 elements
    let keys_prefill = vec![1.0; 16]; // 2 tokens * 8 = 16 elements
    let vals_prefill = vec![2.0; 16];

    cache.append(&keys_prefill, &vals_prefill);
    println!("After prefill: seq_pos={}", cache.seq_pos);
    assert_eq!(cache.seq_pos, 16, "seq_pos should be 16 after prefill");

    // Verify data was written correctly
    assert_eq!(cache.keys[0], 1.0);
    assert_eq!(cache.keys[15], 1.0);
    assert_eq!(cache.values[0], 2.0);

    // Scenario 2: Incremental generation (1 token)
    let keys_inc = vec![3.0; 8]; // 1 token * 8 = 8 elements
    let vals_inc = vec![4.0; 8];

    cache.append(&keys_inc, &vals_inc);
    println!("After incremental: seq_pos={}", cache.seq_pos);
    assert_eq!(cache.seq_pos, 24, "seq_pos should be 24 after incremental");

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
    cache.append(&[1.0; 16], &[2.0; 16]);

    // Get valid portion
    let valid_keys = &cache.keys[..cache.seq_pos];
    assert_eq!(valid_keys.len(), 16, "Should have 16 valid elements");

    // Calculate kv_seq_len (number of tokens)
    let num_kv_heads = 2;
    let head_dim = 4;
    let kv_seq_len = valid_keys.len() / (num_kv_heads * head_dim);
    assert_eq!(kv_seq_len, 2, "Should have 2 tokens in cache");

    println!("✅ KV cache slicing works correctly!");
}
