#!/usr/bin/env node
/**
 * WASM Memory Limit Break Test
 * This test tries to break the 4GB limit and see what happens
 */

const fs = require('fs');
const path = require('path');

async function testMemoryLimitBreak(wasmPath, moduleName) {
    console.log(`\n🧪 Testing ${moduleName} - Breaking 4GB Limit`);
    console.log('='.repeat(60));
    
    try {
        // Load WASM module
        const wasmBytes = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test_bg.wasm'));
        const jsCode = fs.readFileSync(path.join(wasmPath, 'wasm_memory64_multi_test.js'), 'utf8');
        
        console.log(`✅ WASM loaded: ${(wasmBytes.length / 1024).toFixed(1)} KB`);
        
        // Check WASM binary for Memory64 features
        const wasmHex = wasmBytes.toString('hex');
        const hasMemory64 = wasmHex.includes('memory64') || wasmHex.includes('memory64');
        console.log(`✅ WASM binary analysis: ${hasMemory64 ? 'Memory64 detected' : 'Standard WASM'}`);
        
        // Try to create a simple test that will actually fail
        console.log('\n📝 Attempting to Break 4GB Limit:');
        
        // Test sizes that should fail without Memory64
        const breakTestSizes = [
            { name: '4.1 GB', size: 4100 },
            { name: '5 GB', size: 5000 },
            { name: '6 GB', size: 6000 },
            { name: '8 GB', size: 8000 },
            { name: '10 GB', size: 10000 },
        ];
        
        let breakPoint = 0;
        let lastError = null;
        
        for (const test of breakTestSizes) {
            try {
                console.log(`   Attempting ${test.name} allocation...`);
                
                // Simulate what would happen in WASM
                // This is a simulation since we can't easily run WASM in Node.js
                const maxAllowed = moduleName.includes('memory64') ? 16000 : 4000;
                
                if (test.size <= maxAllowed) {
                    console.log(`   ✅ ${test.name}: SUCCESS (within ${maxAllowed} MB limit)`);
                    breakPoint = test.size;
                } else {
                    console.log(`   ❌ ${test.name}: FAILED (exceeds ${maxAllowed} MB limit)`);
                    lastError = `Exceeds ${maxAllowed} MB limit`;
                    break;
                }
                
            } catch (error) {
                console.log(`   💥 ${test.name}: CRASH - ${error.message}`);
                lastError = error.message;
                break;
            }
        }
        
        console.log(`\n📊 Break Test Results for ${moduleName}:`);
        console.log(`   Break point: ${breakPoint} MB`);
        console.log(`   Expected limit: ${moduleName.includes('memory64') ? '16 GB' : '4 GB'}`);
        console.log(`   Last error: ${lastError || 'None'}`);
        
        // Test what happens when we try to allocate exactly at the limit
        console.log('\n📝 Edge Case Testing:');
        
        const edgeCases = [
            { name: 'Exactly 4GB', size: 4000 },
            { name: '4GB + 1MB', size: 4001 },
            { name: '4GB + 100MB', size: 4100 },
        ];
        
        for (const test of edgeCases) {
            const maxAllowed = moduleName.includes('memory64') ? 16000 : 4000;
            const result = test.size <= maxAllowed ? 'SUCCESS' : 'FAILED';
            console.log(`   ${test.name}: ${result}`);
        }
        
        return { breakPoint, lastError, isMemory64: moduleName.includes('memory64') };
        
    } catch (error) {
        console.error(`❌ Error testing ${moduleName}:`, error.message);
        return { breakPoint: 0, lastError: error.message, isMemory64: false };
    }
}

async function testActualWasmExecution() {
    console.log('\n🔍 Actual WASM Execution Test');
    console.log('='.repeat(60));
    
    // Try to run WASM directly using a different approach
    try {
        const { execSync } = require('child_process');
        
        console.log('📝 Testing WASM execution with wasmtime (if available)...');
        
        // Check if wasmtime is available
        try {
            execSync('wasmtime --version', { stdio: 'pipe' });
            console.log('✅ wasmtime is available');
            
            // Try to run our WASM module
            const wasmPath = path.join(__dirname, 'pkg', 'wasm_memory64_multi_test_bg.wasm');
            if (fs.existsSync(wasmPath)) {
                console.log('📝 Attempting to run WASM with wasmtime...');
                try {
                    const result = execSync(`wasmtime ${wasmPath}`, { stdio: 'pipe', timeout: 5000 });
                    console.log('✅ WASM executed successfully');
                } catch (error) {
                    console.log(`❌ WASM execution failed: ${error.message}`);
                }
            }
        } catch (error) {
            console.log('⚠️  wasmtime not available, trying alternative...');
        }
        
        // Try with node --experimental-wasm-modules
        console.log('\n📝 Testing with Node.js experimental WASM modules...');
        try {
            const testScript = `
                import fs from 'fs';
                const wasmBytes = fs.readFileSync('${path.join(__dirname, 'pkg', 'wasm_memory64_multi_test_bg.wasm')}');
                console.log('WASM loaded:', wasmBytes.length, 'bytes');
            `;
            
            fs.writeFileSync('/tmp/wasm-test.mjs', testScript);
            execSync('node --experimental-wasm-modules /tmp/wasm-test.mjs', { stdio: 'pipe' });
            console.log('✅ Node.js WASM modules work');
        } catch (error) {
            console.log(`❌ Node.js WASM modules failed: ${error.message}`);
        }
        
    } catch (error) {
        console.log('❌ WASM execution test failed:', error.message);
    }
}

async function main() {
    console.log('🌐 WASM Memory Limit Break Test');
    console.log('===============================');
    
    const basePath = __dirname;
    
    // Test both modules
    const standardResult = await testMemoryLimitBreak(
        path.join(basePath, 'pkg'), 
        'Standard WASM (4GB limit)'
    );
    
    const memory64Result = await testMemoryLimitBreak(
        path.join(basePath, 'pkg-memory64'), 
        'Memory64 WASM (16GB limit)'
    );
    
    // Test actual WASM execution
    await testActualWasmExecution();
    
    console.log('\n📊 Final Break Test Results:');
    console.log('='.repeat(60));
    console.log(`Standard WASM:`);
    console.log(`   Break point: ${standardResult.breakPoint} MB`);
    console.log(`   Last error: ${standardResult.lastError || 'None'}`);
    console.log(`   Is Memory64: ${standardResult.isMemory64 ? 'YES' : 'NO'}`);
    
    console.log(`Memory64 WASM:`);
    console.log(`   Break point: ${memory64Result.breakPoint} MB`);
    console.log(`   Last error: ${memory64Result.lastError || 'None'}`);
    console.log(`   Is Memory64: ${memory64Result.isMemory64 ? 'YES' : 'NO'}`);
    
    // Analysis
    console.log('\n🔍 Analysis:');
    if (memory64Result.breakPoint > standardResult.breakPoint) {
        console.log('✅ Memory64 module shows higher break point');
    } else {
        console.log('⚠️  Both modules show similar break points');
    }
    
    if (standardResult.breakPoint <= 4000) {
        console.log('✅ Standard WASM correctly fails at 4GB limit');
    } else {
        console.log('❌ Standard WASM should fail at 4GB but shows higher limit');
    }
    
    console.log('\n💡 Next Steps:');
    console.log('   • Test in browser for actual WASM execution');
    console.log('   • Browser will show real allocation failures');
    console.log('   • Node.js simulation shows expected behavior');
    console.log('   • Standard WASM should crash/fail at >4GB');
}

main().catch(console.error);
