#!/usr/bin/env node
/**
 * Node.js WASM Memory Allocation Test
 * This tests actual WASM memory limits by trying to allocate progressively larger amounts
 */

const fs = require('fs');
const path = require('path');

async function testWasmMemoryAllocation(wasmPath, moduleName) {
    console.log(`\n🧪 Testing ${moduleName}`);
    console.log('='.repeat(60));
    
    try {
        // Load WASM module
        const wasmBytes = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test_bg.wasm'));
        const jsCode = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test.js'), 'utf8');
        
        console.log(`✅ WASM loaded: ${(wasmBytes.length / 1024).toFixed(1)} KB`);
        
        // Create a dynamic module loader
        const Module = {
            wasmBinary: wasmBytes,
            print: console.log,
            printErr: console.error,
        };
        
        // Evaluate the JS bindings
        eval(jsCode);
        
        // Wait for WASM to initialize
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Test memory allocation
        console.log('\n📝 Memory Allocation Tests:');
        
        const testSizes = [
            { name: '1 MB', size: 1 },
            { name: '10 MB', size: 10 },
            { name: '100 MB', size: 100 },
            { name: '500 MB', size: 500 },
            { name: '1 GB', size: 1000 },
            { name: '2 GB', size: 2000 },
            { name: '3 GB', size: 3000 },
            { name: '4 GB', size: 4000 },
            { name: '5 GB', size: 5000 },
            { name: '6 GB', size: 6000 },
            { name: '8 GB', size: 8000 },
            { name: '10 GB', size: 10000 },
            { name: '12 GB', size: 12000 },
            { name: '16 GB', size: 16000 },
        ];
        
        let maxSuccessfulAllocation = 0;
        let lastError = null;
        
        for (const test of testSizes) {
            try {
                console.log(`   Testing ${test.name}...`);
                
                // Create a test instance
                const testInstance = new WasmMemoryTest();
                
                // Try to allocate
                const result = testInstance.test_basic_allocation(test.size);
                
                if (result.includes('SUCCESS')) {
                    console.log(`   ✅ ${test.name}: SUCCESS`);
                    maxSuccessfulAllocation = test.size;
                } else {
                    console.log(`   ❌ ${test.name}: ${result}`);
                    lastError = result;
                    break;
                }
                
                // Clean up
                testInstance.free();
                
            } catch (error) {
                console.log(`   ❌ ${test.name}: ERROR - ${error.message}`);
                lastError = error.message;
                break;
            }
        }
        
        console.log(`\n📊 Results for ${moduleName}:`);
        console.log(`   Max successful allocation: ${maxSuccessfulAllocation} MB`);
        console.log(`   Memory limit: ${moduleName.includes('memory64') ? '16 GB' : '4 GB'}`);
        console.log(`   Last error: ${lastError || 'None'}`);
        
        // Test Memory64 specific allocation if available
        if (moduleName.includes('memory64')) {
            console.log('\n📝 Memory64 Specific Tests:');
            try {
                const testInstance = new WasmMemoryTest();
                const result = testInstance.test_memory64_allocation(5000); // 5GB
                console.log(`   Memory64 5GB test: ${result}`);
                testInstance.free();
            } catch (error) {
                console.log(`   Memory64 5GB test: ERROR - ${error.message}`);
            }
        }
        
        return maxSuccessfulAllocation;
        
    } catch (error) {
        console.error(`❌ Error testing ${moduleName}:`, error.message);
        return 0;
    }
}

async function testNodeMemoryLimits() {
    console.log('\n🔍 Node.js Memory Limits Test');
    console.log('='.repeat(60));
    
    // Test Node.js native memory allocation
    const testSizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000];
    
    for (const sizeMB of testSizes) {
        try {
            console.log(`   Testing Node.js ${sizeMB} MB allocation...`);
            const buffer = Buffer.alloc(sizeMB * 1024 * 1024);
            
            // Fill with data to ensure it's actually allocated
            buffer.fill(0x42);
            
            console.log(`   ✅ Node.js ${sizeMB} MB: SUCCESS`);
        } catch (error) {
            console.log(`   ❌ Node.js ${sizeMB} MB: FAILED - ${error.message}`);
            break;
        }
    }
}

async function main() {
    console.log('🌐 WebAssembly Memory Allocation Test');
    console.log('=====================================');
    
    const basePath = __dirname;
    
    // Test Node.js limits first
    await testNodeMemoryLimits();
    
    // Test both WASM modules
    const standardMax = await testWasmMemoryAllocation(
        path.join(basePath, 'pkg'), 
        'Standard WASM (4GB limit)'
    );
    
    const memory64Max = await testWasmMemoryAllocation(
        path.join(basePath, 'pkg-memory64'), 
        'Memory64 WASM (16GB limit)'
    );
    
    console.log('\n📊 Final Results:');
    console.log('='.repeat(60));
    console.log(`Standard WASM max allocation: ${standardMax} MB`);
    console.log(`Memory64 WASM max allocation: ${memory64Max} MB`);
    console.log(`Memory64 improvement: ${memory64Max - standardMax} MB`);
    
    if (memory64Max > standardMax) {
        console.log('✅ Memory64 is working! Higher allocation limit achieved.');
    } else {
        console.log('⚠️  Memory64 may not be working as expected.');
    }
    
    console.log('\n💡 Notes:');
    console.log('   • These are actual WASM memory allocations');
    console.log('   • Results depend on available system memory');
    console.log('   • Memory64 should allow >4GB allocations');
    console.log('   • Node.js may have its own memory limits');
}

main().catch(console.error);
