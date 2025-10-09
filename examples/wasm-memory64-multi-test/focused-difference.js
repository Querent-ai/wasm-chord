#!/usr/bin/env node
/**
 * Focused Memory64 Test - Show the Real Difference
 * This test shows the actual difference between simulated and real WASM memory operations
 */

const fs = require('fs');
const path = require('path');

async function showRealDifference() {
    console.log('🔍 Focused Memory64 Test - Real Difference');
    console.log('==========================================');
    
    const basePath = __dirname;
    
    console.log('\n📝 Standard WASM Module Analysis:');
    console.log('='.repeat(50));
    
    const standardPath = path.join(basePath, 'pkg');
    const standardJsCode = fs.readFileSync(path.join(standardPath, 'wasm_memory64_multi_test.js'), 'utf8');
    
    // Check for actual WASM memory operations
    const standardHasRealMemoryGrow = standardJsCode.includes('const ret = memory.grow(');
    const standardHasRealMemorySize = standardJsCode.includes('const ret = memory.size()');
    
    console.log(`Real memory.grow(): ${standardHasRealMemoryGrow ? '✅ YES' : '❌ NO'}`);
    console.log(`Real memory.size(): ${standardHasRealMemorySize ? '✅ YES' : '❌ NO'}`);
    
    if (!standardHasRealMemoryGrow && !standardHasRealMemorySize) {
        console.log('📝 Standard module uses SIMULATED memory allocation');
        console.log('   • Uses Rust-level memory allocation');
        console.log('   • Limited by Rust memory limits');
        console.log('   • No actual WASM memory operations');
    }
    
    console.log('\n📝 Memory64 WASM Module Analysis:');
    console.log('='.repeat(50));
    
    const memory64Path = path.join(basePath, 'pkg-real-memory64');
    const memory64JsCode = fs.readFileSync(path.join(memory64Path, 'wasm_memory64_multi_test.js'), 'utf8');
    
    // Check for actual WASM memory operations
    const memory64HasRealMemoryGrow = memory64JsCode.includes('const ret = memory.grow(');
    const memory64HasRealMemorySize = memory64JsCode.includes('const ret = memory.size()');
    
    console.log(`Real memory.grow(): ${memory64HasRealMemoryGrow ? '✅ YES' : '❌ NO'}`);
    console.log(`Real memory.size(): ${memory64HasRealMemorySize ? '✅ YES' : '❌ NO'}`);
    
    if (memory64HasRealMemoryGrow && memory64HasRealMemorySize) {
        console.log('📝 Memory64 module uses REAL WASM memory operations');
        console.log('   • Uses actual WASM memory.grow()');
        console.log('   • Uses actual WASM memory.size()');
        console.log('   • Limited by browser/system memory limits');
        console.log('   • Real WebAssembly memory behavior');
    }
    
    console.log('\n🎯 The Key Difference:');
    console.log('='.repeat(50));
    
    if (memory64HasRealMemoryGrow && !standardHasRealMemoryGrow) {
        console.log('✅ Memory64 module uses REAL WASM memory.grow()');
        console.log('✅ Standard module uses SIMULATED memory allocation');
        console.log('');
        console.log('This means:');
        console.log('• Memory64 module will show REAL browser memory limits');
        console.log('• Standard module will show SIMULATED Rust memory limits');
        console.log('• 5GB test should show different results!');
    } else {
        console.log('⚠️  Both modules may have similar memory operations');
        console.log('⚠️  The difference may be in the implementation details');
    }
    
    console.log('\n📊 Expected Test Results:');
    console.log('='.repeat(50));
    
    console.log('5GB Allocation Test:');
    console.log('• Standard WASM (simulated):');
    console.log('  - May succeed or fail depending on Rust memory limits');
    console.log('  - Uses Rust-level memory allocation');
    console.log('  - Not limited by WASM 4GB limit');
    console.log('');
    console.log('• Memory64 WASM (real):');
    console.log('  - Uses actual WASM memory.grow()');
    console.log('  - Limited by browser/system memory');
    console.log('  - Shows real WebAssembly memory behavior');
    console.log('  - May succeed if browser supports >4GB');
    
    console.log('\n🌐 Browser Testing:');
    console.log('='.repeat(50));
    console.log('• Open: http://localhost:8001/comparison-test.html');
    console.log('• Click "Run 5GB Allocation Test"');
    console.log('• The results will show the real difference!');
    
    console.log('\n💡 What to Look For:');
    console.log('='.repeat(50));
    console.log('• Different error messages');
    console.log('• Different memory statistics');
    console.log('• Different allocation behavior');
    console.log('• Real vs simulated memory limits');
    
    return {
        standardHasRealMemoryGrow,
        memory64HasRealMemoryGrow,
        standardHasRealMemorySize,
        memory64HasRealMemorySize,
        realDifference: memory64HasRealMemoryGrow && !standardHasRealMemoryGrow
    };
}

async function main() {
    const results = await showRealDifference();
    
    console.log('\n🎉 Analysis Complete!');
    console.log('='.repeat(50));
    
    if (results.realDifference) {
        console.log('✅ REAL difference found!');
        console.log('✅ Memory64 uses actual WASM memory operations');
        console.log('✅ Standard uses simulated memory allocation');
        console.log('✅ 5GB test should show different results');
    } else {
        console.log('⚠️  Similar memory operations found');
        console.log('⚠️  Difference may be in implementation details');
    }
    
    console.log('\nNext: Test in browser to see the real difference!');
    console.log('URL: http://localhost:8001/comparison-test.html');
}

main().catch(console.error);
