#!/usr/bin/env node
/**
 * Node.js test script for wasm-capital-test
 *
 * Tests the WASM module in Node.js environment
 *
 * Usage:
 *   node test.js [path-to-model.gguf]
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main() {
    console.log('🧪 WASM Capital Test (Node.js)\n');

    // Get model path from args or use default
    const modelPath = process.argv[2] ||
        process.env.WASM_CHORD_TEST_MODEL ||
        join(process.env.HOME, '.ollama', 'models', 'tinyllama-1.1b.Q4_K_M.gguf');

    console.log(`📦 Model path: ${modelPath}\n`);

    try {
        // Import WASM module
        console.log('🔧 Loading WASM module...');
        const wasmModule = await import('./pkg/wasm_capital_test.js');

        // Load WASM binary and initialize
        const wasmPath = join(__dirname, 'pkg', 'wasm_capital_test_bg.wasm');
        const wasmBytes = await readFile(wasmPath);
        await wasmModule.default(wasmBytes); // Initialize with WASM bytes
        console.log('✅ WASM initialized\n');

        // Get module info
        const info = wasmModule.get_test_info();
        console.log(`ℹ️  ${info}\n`);

        // Load model file
        console.log('📥 Loading model file...');
        const modelBytes = await readFile(modelPath);
        console.log(`✅ Loaded ${modelBytes.length} bytes\n`);

        // Run inference test
        console.log('🚀 Running inference test...');
        console.log('   Prompt: "What is the capital of France?"');
        console.log('   Expected: Response containing "Paris"\n');

        const startTime = Date.now();
        const result = await wasmModule.test_capital_inference(modelBytes);
        const elapsed = Date.now() - startTime;

        // Display results
        console.log('📝 Results:');
        console.log(`   Prompt: ${result.prompt}`);
        console.log(`   Response: ${result.response}`);
        console.log(`   Time: ${(elapsed / 1000).toFixed(2)}s\n`);

        if (result.success) {
            console.log('✅ SUCCESS: Model correctly identified Paris!');
            process.exit(0);
        } else {
            console.log('❌ FAILED: Expected "Paris" in response');
            process.exit(1);
        }

    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('\nStack trace:', error.stack);
        process.exit(1);
    }
}

main();
