#!/usr/bin/env node
/**
 * Conservative WASM Memory Test
 * Tests WASM memory limits without crashing the system
 */

const fs = require('fs');
const path = require('path');

async function testWasmMemoryConservative(wasmPath, moduleName) {
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
        
        // Test memory allocation with conservative limits
        console.log('\n📝 WASM Memory Allocation Tests:');
        
        const testSizes = [
            { name: '1 MB', size: 1 },
            { name: '10 MB', size: 10 },
            { name: '50 MB', size: 50 },
            { name: '100 MB', size: 100 },
            { name: '200 MB', size: 200 },
            { name: '500 MB', size: 500 },
            { name: '1 GB', size: 1000 },
            { name: '1.5 GB', size: 1500 },
            { name: '2 GB', size: 2000 },
            { name: '2.5 GB', size: 2500 },
            { name: '3 GB', size: 3000 },
            { name: '3.5 GB', size: 3500 },
            { name: '4 GB', size: 4000 },
            { name: '4.5 GB', size: 4500 },
            { name: '5 GB', size: 5000 },
        ];
        
        let maxSuccessfulAllocation = 0;
        let lastError = null;
        let successCount = 0;
        
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
                    successCount++;
                } else {
                    console.log(`   ❌ ${test.name}: ${result}`);
                    lastError = result;
                    break;
                }
                
                // Clean up
                testInstance.free();
                
                // Small delay to prevent overwhelming the system
                await new Promise(resolve => setTimeout(resolve, 100));
                
            } catch (error) {
                console.log(`   ❌ ${test.name}: ERROR - ${error.message}`);
                lastError = error.message;
                break;
            }
        }
        
        console.log(`\n📊 Results for ${moduleName}:`);
        console.log(`   Max successful allocation: ${maxSuccessfulAllocation} MB`);
        console.log(`   Successful tests: ${successCount}`);
        console.log(`   Memory limit: ${moduleName.includes('memory64') ? '16 GB' : '4 GB'}`);
        console.log(`   Last error: ${lastError || 'None'}`);
        
        // Test Memory64 specific allocation if available
        if (moduleName.includes('memory64')) {
            console.log('\n📝 Memory64 Specific Tests:');
            try {
                const testInstance = new WasmMemoryTest();
                const result = testInstance.test_memory64_allocation(2000); // 2GB
                console.log(`   Memory64 2GB test: ${result}`);
                testInstance.free();
            } catch (error) {
                console.log(`   Memory64 2GB test: ERROR - ${error.message}`);
            }
        }
        
        return { maxAllocation: maxSuccessfulAllocation, successCount };
        
    } catch (error) {
        console.error(`❌ Error testing ${moduleName}:`, error.message);
        return { maxAllocation: 0, successCount: 0 };
    }
}

async function testSystemMemory() {
    console.log('\n🔍 System Memory Information');
    console.log('='.repeat(60));
    
    try {
        const os = require('os');
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        const usedMem = totalMem - freeMem;
        
        console.log(`Total system memory: ${(totalMem / 1024 / 1024 / 1024).toFixed(2)} GB`);
        console.log(`Free system memory: ${(freeMem / 1024 / 1024 / 1024).toFixed(2)} GB`);
        console.log(`Used system memory: ${(usedMem / 1024 / 1024 / 1024).toFixed(2)} GB`);
        console.log(`Memory usage: ${((usedMem / totalMem) * 100).toFixed(1)}%`);
        
        // Test small Node.js allocation
        console.log('\n📝 Node.js Small Allocation Test:');
        try {
            const buffer = Buffer.alloc(100 * 1024 * 1024); // 100MB
            buffer.fill(0x42);
            console.log('   ✅ Node.js 100MB: SUCCESS');
        } catch (error) {
            console.log(`   ❌ Node.js 100MB: FAILED - ${error.message}`);
        }
        
    } catch (error) {
        console.error('❌ Error getting system memory info:', error.message);
    }
}

async function main() {
    console.log('🌐 Conservative WebAssembly Memory Test');
    console.log('=======================================');
    
    const basePath = __dirname;
    
    // Test system memory first
    await testSystemMemory();
    
    // Test both WASM modules
    const standardResult = await testWasmMemoryConservative(
        path.join(basePath, 'pkg'), 
        'Standard WASM (4GB limit)'
    );
    
    const memory64Result = await testWasmMemoryConservative(
        path.join(basePath, 'pkg-memory64'), 
        'Memory64 WASM (16GB limit)'
    );
    
    console.log('\n📊 Final Comparison:');
    console.log('='.repeat(60));
    console.log(`Standard WASM max allocation: ${standardResult.maxAllocation} MB`);
    console.log(`Memory64 WASM max allocation: ${memory64Result.maxAllocation} MB`);
    console.log(`Standard WASM successful tests: ${standardResult.successCount}`);
    console.log(`Memory64 WASM successful tests: ${memory64Result.successCount}`);
    
    if (memory64Result.maxAllocation > standardResult.maxAllocation) {
        console.log('✅ Memory64 is working! Higher allocation limit achieved.');
        console.log(`   Improvement: ${memory64Result.maxAllocation - standardResult.maxAllocation} MB`);
    } else if (memory64Result.maxAllocation === standardResult.maxAllocation) {
        console.log('⚠️  Memory64 and Standard show same limits (may be system limited)');
    } else {
        console.log('❌ Memory64 may not be working as expected.');
    }
    
    console.log('\n💡 Notes:');
    console.log('   • These are actual WASM memory allocations');
    console.log('   • Results are limited by available system memory');
    console.log('   • Memory64 should allow >4GB allocations when system has enough RAM');
    console.log('   • Previous test was killed at 8GB due to system memory limits');
}

main().catch(console.error);
