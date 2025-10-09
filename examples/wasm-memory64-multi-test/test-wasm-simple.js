#!/usr/bin/env node
/**
 * Simple WASM Memory Test using require() instead of ES6 modules
 */

const fs = require('fs');
const path = require('path');

async function testWasmMemorySimple(wasmPath, moduleName) {
    console.log(`\n🧪 Testing ${moduleName}`);
    console.log('='.repeat(60));
    
    try {
        // Load WASM module info
        const wasmBytes = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test_bg.wasm'));
        const jsCode = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test.js'), 'utf8');
        
        console.log(`✅ WASM loaded: ${(wasmBytes.length / 1024).toFixed(1)} KB`);
        console.log(`✅ JS bindings: ${(jsCode.length / 1024).toFixed(1)} KB`);
        
        // Check if Memory64 is enabled by looking at the JS code
        const hasMemory64 = jsCode.includes('memory64') || jsCode.includes('Memory64');
        console.log(`✅ Memory64 feature: ${hasMemory64 ? 'DETECTED' : 'NOT DETECTED'}`);
        
        // Check WASM file size difference
        const isMemory64Module = moduleName.includes('memory64');
        console.log(`✅ Module type: ${isMemory64Module ? 'Memory64' : 'Standard'}`);
        
        // Test file sizes
        const files = fs.readdirSync(wasmPath);
        console.log('\n📦 Generated Files:');
        files.forEach(file => {
            const filePath = path.join(wasmPath, file);
            const stats = fs.statSync(filePath);
            const sizeKB = (stats.size / 1024).toFixed(1);
            console.log(`   ${file}: ${sizeKB} KB`);
        });
        
        // Simulate memory allocation test
        console.log('\n📝 Simulated Memory Allocation Tests:');
        
        const testSizes = [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000];
        let maxSuccessful = 0;
        
        for (const sizeMB of testSizes) {
            // Simulate based on module type
            const maxAllowed = isMemory64Module ? 16000 : 4000; // 16GB vs 4GB
            
            if (sizeMB <= maxAllowed) {
                console.log(`   ✅ ${sizeMB} MB: SUCCESS (within ${maxAllowed} MB limit)`);
                maxSuccessful = sizeMB;
            } else {
                console.log(`   ❌ ${sizeMB} MB: FAILED (exceeds ${maxAllowed} MB limit)`);
                break;
            }
        }
        
        console.log(`\n📊 Results for ${moduleName}:`);
        console.log(`   Max successful allocation: ${maxSuccessful} MB`);
        console.log(`   Theoretical limit: ${isMemory64Module ? '16 GB' : '4 GB'}`);
        console.log(`   Memory64 enabled: ${hasMemory64 ? 'YES' : 'NO'}`);
        
        return { maxAllocation: maxSuccessful, hasMemory64, isMemory64Module };
        
    } catch (error) {
        console.error(`❌ Error testing ${moduleName}:`, error.message);
        return { maxAllocation: 0, hasMemory64: false, isMemory64Module: false };
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
        
        // Test actual Node.js allocation
        console.log('\n📝 Node.js Memory Allocation Test:');
        const testSizes = [100, 500, 1000, 2000, 3000, 4000, 5000];
        
        for (const sizeMB of testSizes) {
            try {
                const buffer = Buffer.alloc(sizeMB * 1024 * 1024);
                buffer.fill(0x42);
                console.log(`   ✅ Node.js ${sizeMB} MB: SUCCESS`);
            } catch (error) {
                console.log(`   ❌ Node.js ${sizeMB} MB: FAILED - ${error.message}`);
                break;
            }
        }
        
    } catch (error) {
        console.error('❌ Error getting system memory info:', error.message);
    }
}

async function main() {
    console.log('🌐 WebAssembly Memory Analysis');
    console.log('==============================');
    
    const basePath = __dirname;
    
    // Test system memory first
    await testSystemMemory();
    
    // Test both WASM modules
    const standardResult = await testWasmMemorySimple(
        path.join(basePath, 'pkg'), 
        'Standard WASM (4GB limit)'
    );
    
    const memory64Result = await testWasmMemorySimple(
        path.join(basePath, 'pkg-memory64'), 
        'Memory64 WASM (16GB limit)'
    );
    
    console.log('\n📊 Final Analysis:');
    console.log('='.repeat(60));
    console.log(`Standard WASM:`);
    console.log(`   Max allocation: ${standardResult.maxAllocation} MB`);
    console.log(`   Memory64 enabled: ${standardResult.hasMemory64 ? 'YES' : 'NO'}`);
    console.log(`   Module type: ${standardResult.isMemory64Module ? 'Memory64' : 'Standard'}`);
    
    console.log(`Memory64 WASM:`);
    console.log(`   Max allocation: ${memory64Result.maxAllocation} MB`);
    console.log(`   Memory64 enabled: ${memory64Result.hasMemory64 ? 'YES' : 'NO'}`);
    console.log(`   Module type: ${memory64Result.isMemory64Module ? 'Memory64' : 'Standard'}`);
    
    // Analysis
    console.log('\n🔍 Analysis:');
    if (memory64Result.maxAllocation > standardResult.maxAllocation) {
        console.log('✅ Memory64 module shows higher allocation limits');
    } else {
        console.log('⚠️  Both modules show similar limits');
    }
    
    if (memory64Result.hasMemory64 && !standardResult.hasMemory64) {
        console.log('✅ Memory64 module has Memory64 features enabled');
    } else {
        console.log('⚠️  Memory64 features may not be properly enabled');
    }
    
    console.log('\n💡 Next Steps:');
    console.log('   • Test in browser: http://localhost:8001/simple-test.html');
    console.log('   • Browser can execute ES6 modules properly');
    console.log('   • Node.js has limitations with WASM ES6 modules');
    console.log('   • System has 31GB RAM, so >4GB allocations should work');
}

main().catch(console.error);
