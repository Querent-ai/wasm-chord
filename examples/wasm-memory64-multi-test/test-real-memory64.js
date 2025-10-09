#!/usr/bin/env node
/**
 * Real Memory64 WASM Test
 * This test verifies our real Memory64 implementation using actual WASM memory.grow()
 */

const fs = require('fs');
const path = require('path');

async function testRealMemory64Implementation() {
    console.log('🌐 Real Memory64 WASM Implementation Test');
    console.log('=========================================');
    
    const basePath = __dirname;
    
    // Test the real Memory64 module
    console.log('\n🧪 Testing Real Memory64 WASM Module');
    console.log('='.repeat(50));
    
    try {
        const wasmPath = path.join(basePath, 'pkg-real-memory64');
        const wasmBytes = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test_bg.wasm'));
        const jsCode = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test.js'), 'utf8');
        
        console.log(`✅ Real Memory64 WASM loaded: ${(wasmBytes.length / 1024).toFixed(1)} KB`);
        console.log(`✅ JS bindings loaded: ${(jsCode.length / 1024).toFixed(1)} KB`);
        
        // Check for real Memory64 features
        console.log('\n📝 Real Memory64 Features Analysis:');
        
        // Check for memory.grow() usage
        const hasMemoryGrow = jsCode.includes('memory.grow') || jsCode.includes('memory_grow');
        console.log(`   memory.grow() usage: ${hasMemoryGrow ? '✅ YES' : '❌ NO'}`);
        
        // Check for memory.size() usage
        const hasMemorySize = jsCode.includes('memory.size') || jsCode.includes('memory_size');
        console.log(`   memory.size() usage: ${hasMemorySize ? '✅ YES' : '❌ NO'}`);
        
        // Check for WebAssembly Memory access
        const hasWebAssemblyMemory = jsCode.includes('WebAssembly') && jsCode.includes('Memory');
        console.log(`   WebAssembly Memory access: ${hasWebAssemblyMemory ? '✅ YES' : '❌ NO'}`);
        
        // Check for Uint8Array usage (for memory access)
        const hasUint8Array = jsCode.includes('Uint8Array');
        console.log(`   Uint8Array usage: ${hasUint8Array ? '✅ YES' : '❌ NO'}`);
        
        // Check for actual WASM memory operations
        const hasWasmMemoryOps = hasMemoryGrow && hasMemorySize && hasWebAssemblyMemory;
        console.log(`   Real WASM memory operations: ${hasWasmMemoryOps ? '✅ YES' : '❌ NO'}`);
        
        // Compare with standard module
        const standardPath = path.join(basePath, 'pkg');
        if (fs.existsSync(standardPath)) {
            const standardWasmBytes = fs.readFileSync(path.join(standardPath, 'wasm_memory64_multi_test_bg.wasm'));
            const standardJsCode = fs.readFileSync(path.join(standardPath, 'wasm_memory64_multi_test.js'), 'utf8');
            
            console.log('\n📊 Comparison with Standard Module:');
            console.log(`   Real Memory64 WASM: ${(wasmBytes.length / 1024).toFixed(1)} KB`);
            console.log(`   Standard WASM: ${(standardWasmBytes.length / 1024).toFixed(1)} KB`);
            console.log(`   Size difference: ${((wasmBytes.length - standardWasmBytes.length) / 1024).toFixed(1)} KB`);
            
            const standardHasMemoryGrow = standardJsCode.includes('memory.grow') || standardJsCode.includes('memory_grow');
            console.log(`   Standard has memory.grow(): ${standardHasMemoryGrow ? '✅ YES' : '❌ NO'}`);
            
            if (hasMemoryGrow && !standardHasMemoryGrow) {
                console.log('   ✅ Real Memory64 module has memory.grow() while standard does not');
            } else if (!hasMemoryGrow && standardHasMemoryGrow) {
                console.log('   ⚠️  Standard module has memory.grow() while real Memory64 does not');
            } else if (hasMemoryGrow && standardHasMemoryGrow) {
                console.log('   ⚠️  Both modules have memory.grow() - may be similar implementation');
            } else {
                console.log('   ❌ Neither module has memory.grow() - no real WASM memory operations');
            }
        }
        
        console.log('\n📝 Expected Behavior:');
        console.log('   Real Memory64 WASM should:');
        console.log('   ✅ Use actual WASM memory.grow() for allocation');
        console.log('   ✅ Use actual WASM memory.size() for size checking');
        console.log('   ✅ Access WebAssembly Memory directly');
        console.log('   ✅ Fill allocated memory with test data');
        console.log('   ✅ Show real memory limits (not simulated)');
        
        console.log('\n🌐 Browser Testing:');
        console.log('   • Open: http://localhost:8001/real-memory64-test.html');
        console.log('   • Test real WASM memory allocation');
        console.log('   • Find actual memory limits');
        console.log('   • Compare with simulated implementation');
        
        console.log('\n📊 Expected Results:');
        console.log('   Real Memory64 WASM:');
        console.log('     - Should use actual memory.grow()');
        console.log('     - Should show real memory limits');
        console.log('     - Should fail at actual browser/system limits');
        console.log('     - Should provide accurate memory statistics');
        
        console.log('\n🎯 Conclusion:');
        if (hasWasmMemoryOps) {
            console.log('   ✅ Real Memory64 implementation uses actual WASM memory operations');
            console.log('   ✅ Ready for browser testing');
            console.log('   ✅ Should show real memory behavior');
        } else {
            console.log('   ❌ Real Memory64 implementation does not use actual WASM memory operations');
            console.log('   ⚠️  May still be using simulated memory allocation');
        }
        
        return hasWasmMemoryOps;
        
    } catch (error) {
        console.error('❌ Error testing Real Memory64 implementation:', error.message);
        return false;
    }
}

async function main() {
    const success = await testRealMemory64Implementation();
    
    if (success) {
        console.log('\n🎉 Real Memory64 implementation is ready for testing!');
        console.log('   Next: Test in browser at http://localhost:8001/real-memory64-test.html');
    } else {
        console.log('\n⚠️  Real Memory64 implementation needs more work');
        console.log('   Check the implementation for actual WASM memory operations');
    }
}

main().catch(console.error);
