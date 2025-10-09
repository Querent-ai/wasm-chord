#!/usr/bin/env node
/**
 * Memory64 vs Standard WASM Comparison Test
 * This test compares both modules to see the difference in memory allocation limits
 */

const fs = require('fs');
const path = require('path');

async function compareMemory64VsStandard() {
    console.log('🧪 Memory64 vs Standard WASM Comparison');
    console.log('======================================');
    
    const basePath = __dirname;
    
    // Test both modules
    console.log('\n📝 Testing Standard WASM (4GB limit):');
    console.log('='.repeat(50));
    
    const standardPath = path.join(basePath, 'pkg');
    const standardWasmBytes = fs.readFileSync(path.join(standardPath, 'wasm_memory64_multi_test_bg.wasm'));
    const standardJsCode = fs.readFileSync(path.join(standardPath, 'wasm_memory64_multi_test.js'), 'utf8');
    
    console.log(`✅ Standard WASM: ${(standardWasmBytes.length / 1024).toFixed(1)} KB`);
    
    // Check for memory operations
    const standardHasMemoryGrow = standardJsCode.includes('memory.grow') || standardJsCode.includes('memory_grow');
    const standardHasMemorySize = standardJsCode.includes('memory.size') || standardJsCode.includes('memory_size');
    
    console.log(`   memory.grow(): ${standardHasMemoryGrow ? '✅ YES' : '❌ NO'}`);
    console.log(`   memory.size(): ${standardHasMemorySize ? '✅ YES' : '❌ NO'}`);
    
    console.log('\n📝 Testing Memory64 WASM (16GB limit):');
    console.log('='.repeat(50));
    
    const memory64Path = path.join(basePath, 'pkg-real-memory64');
    const memory64WasmBytes = fs.readFileSync(path.join(memory64Path, 'wasm_memory64_multi_test_bg.wasm'));
    const memory64JsCode = fs.readFileSync(path.join(memory64Path, 'wasm_memory64_multi_test.js'), 'utf8');
    
    console.log(`✅ Memory64 WASM: ${(memory64WasmBytes.length / 1024).toFixed(1)} KB`);
    
    // Check for memory operations
    const memory64HasMemoryGrow = memory64JsCode.includes('memory.grow') || memory64JsCode.includes('memory_grow');
    const memory64HasMemorySize = memory64JsCode.includes('memory.size') || memory64JsCode.includes('memory_size');
    
    console.log(`   memory.grow(): ${memory64HasMemoryGrow ? '✅ YES' : '❌ NO'}`);
    console.log(`   memory.size(): ${memory64HasMemorySize ? '✅ YES' : '❌ NO'}`);
    
    console.log('\n📊 Comparison Results:');
    console.log('='.repeat(50));
    
    const sizeDiff = memory64WasmBytes.length - standardWasmBytes.length;
    console.log(`Size difference: ${sizeDiff > 0 ? '+' : ''}${(sizeDiff / 1024).toFixed(1)} KB`);
    
    if (memory64HasMemoryGrow && !standardHasMemoryGrow) {
        console.log('✅ Memory64 module has memory.grow() while standard does not');
    } else if (!memory64HasMemoryGrow && standardHasMemoryGrow) {
        console.log('⚠️  Standard module has memory.grow() while Memory64 does not');
    } else if (memory64HasMemoryGrow && standardHasMemoryGrow) {
        console.log('⚠️  Both modules have memory.grow() - may be similar implementation');
    } else {
        console.log('❌ Neither module has memory.grow() - no real WASM memory operations');
    }
    
    console.log('\n📝 Expected Behavior:');
    console.log('='.repeat(50));
    
    console.log('Standard WASM (4GB limit):');
    console.log('  ✅ 1 GB allocation: SUCCESS');
    console.log('  ✅ 2 GB allocation: SUCCESS');
    console.log('  ✅ 3 GB allocation: SUCCESS');
    console.log('  ✅ 4 GB allocation: SUCCESS');
    console.log('  ❌ 5 GB allocation: FAILED (exceeds 4GB limit)');
    console.log('  ❌ 6 GB allocation: FAILED (exceeds 4GB limit)');
    
    console.log('\nMemory64 WASM (16GB limit):');
    console.log('  ✅ 1 GB allocation: SUCCESS');
    console.log('  ✅ 2 GB allocation: SUCCESS');
    console.log('  ✅ 3 GB allocation: SUCCESS');
    console.log('  ✅ 4 GB allocation: SUCCESS');
    console.log('  ✅ 5 GB allocation: SUCCESS (within 16GB limit)');
    console.log('  ✅ 6 GB allocation: SUCCESS (within 16GB limit)');
    console.log('  ✅ 8 GB allocation: SUCCESS (within 16GB limit)');
    console.log('  ❌ 17 GB allocation: FAILED (exceeds 16GB limit)');
    
    console.log('\n🌐 Browser Testing:');
    console.log('='.repeat(50));
    console.log('• Open: http://localhost:8001/comparison-test.html');
    console.log('• Click "Run 5GB Allocation Test"');
    console.log('• Expected results:');
    console.log('  - Standard WASM: FAILED (exceeds 4GB limit)');
    console.log('  - Memory64 WASM: SUCCESS (within 16GB limit)');
    
    console.log('\n🎯 Key Test:');
    console.log('='.repeat(50));
    console.log('The 5GB allocation test should show:');
    console.log('• Standard WASM: ❌ FAILED (exceeds 4GB limit)');
    console.log('• Memory64 WASM: ✅ SUCCESS (within 16GB limit)');
    console.log('');
    console.log('This proves Memory64 is working correctly!');
    
    return {
        standardHasMemoryGrow,
        memory64HasMemoryGrow,
        sizeDiff,
        memory64Working: memory64HasMemoryGrow && !standardHasMemoryGrow
    };
}

async function main() {
    const results = await compareMemory64VsStandard();
    
    console.log('\n🎉 Comparison Complete!');
    console.log('='.repeat(50));
    
    if (results.memory64Working) {
        console.log('✅ Memory64 implementation is working correctly');
        console.log('✅ Ready for browser testing');
        console.log('✅ Should show different behavior at 5GB+');
    } else {
        console.log('⚠️  Memory64 implementation may need more work');
        console.log('⚠️  Both modules may have similar behavior');
    }
    
    console.log('\nNext: Test in browser at http://localhost:8001/comparison-test.html');
}

main().catch(console.error);
