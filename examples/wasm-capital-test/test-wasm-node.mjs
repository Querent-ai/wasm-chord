#!/usr/bin/env node
/**
 * WASM Runtime Test for Node.js / CI
 *
 * Tests:
 * - WASM module loads correctly
 * - Model initialization works
 * - Generate produces correct output (contains "Paris")
 * - Async API functions work
 * - Performance benchmarks
 */

import { readFileSync, existsSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const MODEL_PATH = join(__dirname, 'models', 'tinyllama-1.1b.Q4_K_M.gguf');
const WASM_PKG_PATH = join(__dirname, 'runtime-pkg-node');
const TEST_PROMPT = "What is the capital of France?";
const EXPECTED_ANSWER = "paris"; // lowercase for case-insensitive check
const MAX_TOKENS = 20;
const TEMPERATURE = 0.0; // Greedy for deterministic output

// Test results
const results = {
    passed: 0,
    failed: 0,
    tests: [],
    benchmarks: {}
};

function logTest(name, passed, message) {
    const status = passed ? '✅ PASS' : '❌ FAIL';
    console.log(`${status}: ${name}`);
    if (message) {
        console.log(`   ${message}`);
    }

    results.tests.push({ name, passed, message });
    if (passed) {
        results.passed++;
    } else {
        results.failed++;
    }
}

function logBenchmark(name, value, unit) {
    console.log(`📊 ${name}: ${value}${unit}`);
    results.benchmarks[name] = { value, unit };
}

async function main() {
    console.log('🧪 WASM Runtime CI Test');
    console.log('═'.repeat(60));
    console.log('');

    // Test 1: Check if model file exists
    console.log('📦 Test 1: Model File');
    const modelExists = existsSync(MODEL_PATH);
    const modelStats = modelExists ? statSync(MODEL_PATH) : null;
    logTest(
        'Model file exists',
        modelExists,
        modelExists ? `Size: ${(modelStats.size / 1024 / 1024).toFixed(2)} MB` : `Not found at ${MODEL_PATH}`
    );

    if (!modelExists) {
        console.log('');
        console.log('❌ Cannot proceed without model file');
        console.log('   The symlink should already exist. Check:');
        console.log(`   ls -la ${MODEL_PATH}`);
        printSummary();
        process.exit(1);
    }

    // Test 2: Check WASM package exists
    console.log('');
    console.log('📦 Test 2: WASM Package');
    const wasmJsPath = join(WASM_PKG_PATH, 'wasm_chord_runtime.js');
    const wasmBinPath = join(WASM_PKG_PATH, 'wasm_chord_runtime_bg.wasm');
    const wasmJsExists = existsSync(wasmJsPath);
    const wasmBinExists = existsSync(wasmBinPath);

    logTest('WASM JS file exists', wasmJsExists, wasmJsExists ? 'Found' : 'Not found');
    logTest('WASM binary exists', wasmBinExists, wasmBinExists ? 'Found' : 'Not found');

    if (!wasmJsExists || !wasmBinExists) {
        console.log('');
        console.log('❌ WASM package not found');
        console.log('   It should have been built. Check:');
        console.log(`   ls -la ${WASM_PKG_PATH}`);
        printSummary();
        process.exit(1);
    }

    // Test 3: Load WASM module
    console.log('');
    console.log('🔧 Test 3: Load WASM Module');
    let WasmModel;
    try {
        const wasm = await import(wasmJsPath);
        WasmModel = wasm.WasmModel;
        logTest('WASM module load', true, 'Successfully imported');
        logTest('WasmModel class available', typeof WasmModel === 'function', 'WasmModel constructor found');
    } catch (error) {
        logTest('WASM module load', false, error.message);
        printSummary();
        process.exit(1);
    }

    // Test 4: Load model file
    console.log('');
    console.log('📥 Test 4: Load Model');
    let modelBytes;
    try {
        const loadStart = Date.now();
        modelBytes = readFileSync(MODEL_PATH);
        const loadTime = Date.now() - loadStart;
        logTest('Model file read', true, `${(modelBytes.length / 1024 / 1024).toFixed(2)} MB`);
        logBenchmark('Model load time', loadTime, 'ms');
    } catch (error) {
        logTest('Model file read', false, error.message);
        printSummary();
        process.exit(1);
    }

    // Test 5: Initialize WASM model
    console.log('');
    console.log('🏗️  Test 5: Initialize Model');
    let model;
    try {
        const initStart = Date.now();
        model = new WasmModel(modelBytes);
        const initTime = Date.now() - initStart;
        logTest('Model initialization', true, 'Model created successfully');
        logBenchmark('Model init time', initTime, 'ms');

        // Configure generation
        model.set_config(MAX_TOKENS, TEMPERATURE, 1.0, 0, 1.0);
        logTest('Model configuration', true, `max_tokens=${MAX_TOKENS}, temp=${TEMPERATURE}`);
    } catch (error) {
        logTest('Model initialization', false, error.message);
        console.error('Stack trace:', error.stack);
        printSummary();
        process.exit(1);
    }

    // Test 6: Check model info
    console.log('');
    console.log('ℹ️  Test 6: Model Info');
    try {
        const info = model.get_model_info();
        logTest('Get model info', true, `vocab_size=${info.vocab_size}, hidden_size=${info.hidden_size}`);
        console.log(`   Layers: ${info.num_layers}, Heads: ${info.num_heads}, Max seq: ${info.max_seq_len}`);
    } catch (error) {
        logTest('Get model info', false, error.message);
    }

    // Test 7: Synchronous generation
    console.log('');
    console.log('🤖 Test 7: Synchronous Generation');
    let response;
    try {
        const genStart = Date.now();
        response = model.generate(TEST_PROMPT);
        const genTime = Date.now() - genStart;

        logTest('Generate (sync)', true, 'Generation completed');
        logBenchmark('Generation time', genTime, 'ms');
        logBenchmark('Tokens per second', ((MAX_TOKENS / (genTime / 1000)).toFixed(2)), ' tok/s');

        console.log(`   Prompt: "${TEST_PROMPT}"`);
        console.log(`   Response: "${response}"`);

        // Test 8: Verify output contains "Paris"
        console.log('');
        console.log('✓ Test 8: Output Verification');
        const containsParis = response.toLowerCase().includes(EXPECTED_ANSWER);
        logTest(
            'Response contains "Paris"',
            containsParis,
            containsParis ? 'Output is correct ✓' : `Expected "${EXPECTED_ANSWER}" in response`
        );

        if (!containsParis) {
            console.log('   ⚠️  This may indicate a problem with model inference');
        }
    } catch (error) {
        logTest('Generate (sync)', false, error.message);
        console.error('Stack trace:', error.stack);
    }

    // Test 9: Streaming generation
    console.log('');
    console.log('🌊 Test 9: Streaming Generation');
    try {
        let streamedText = '';
        let tokenCount = 0;

        const streamStart = Date.now();
        const streamResponse = model.generate_stream(TEST_PROMPT, (token) => {
            streamedText += token;
            tokenCount++;
            return true; // continue
        });
        const streamTime = Date.now() - streamStart;

        logTest('Generate (stream)', true, `Received ${tokenCount} tokens`);
        logBenchmark('Stream time', streamTime, 'ms');
        console.log(`   Streamed: "${streamedText}"`);

        // Verify streaming output
        const streamMatch = streamedText.toLowerCase().includes(EXPECTED_ANSWER);
        logTest('Streaming output matches', streamMatch, streamMatch ? 'Consistent ✓' : 'Differs from sync');
    } catch (error) {
        logTest('Generate (stream)', false, error.message);
    }

    // Test 10: Async generation
    console.log('');
    console.log('⚡ Test 10: Async Generation');
    try {
        const asyncStart = Date.now();
        const stream = model.generate_async(TEST_PROMPT);

        logTest('Async stream creation', true, 'AsyncTokenStream created');

        // Get first token
        const firstToken = await stream.next();
        const firstTokenTime = Date.now() - asyncStart;

        logTest('Async first token', firstToken && !firstToken.done, `Got: "${firstToken.value}"`);
        logBenchmark('Time to first token', firstTokenTime, 'ms');

        // Get a few more tokens to verify iterator works
        let asyncText = firstToken.value || '';
        let asyncCount = 1;

        for (let i = 0; i < 5; i++) {
            const token = await stream.next();
            if (token.done) break;
            asyncText += token.value;
            asyncCount++;
        }

        const asyncTime = Date.now() - asyncStart;
        logTest('Async iteration', true, `Retrieved ${asyncCount} tokens`);
        logBenchmark('Async generation time', asyncTime, 'ms');
        console.log(`   Async text: "${asyncText}"`);

    } catch (error) {
        logTest('Async generation', false, error.message);
        console.error('Stack trace:', error.stack);
    }

    // Print final summary
    console.log('');
    printSummary();

    // Exit with appropriate code
    process.exit(results.failed > 0 ? 1 : 0);
}

function printSummary() {
    console.log('═'.repeat(60));
    console.log('📊 Test Summary');
    console.log('═'.repeat(60));
    console.log(`Total Tests: ${results.tests.length}`);
    console.log(`✅ Passed: ${results.passed}`);
    console.log(`❌ Failed: ${results.failed}`);

    if (Object.keys(results.benchmarks).length > 0) {
        console.log('');
        console.log('⏱️  Benchmarks:');
        Object.entries(results.benchmarks).forEach(([name, data]) => {
            console.log(`   ${name}: ${data.value}${data.unit}`);
        });
    }

    if (results.failed > 0) {
        console.log('');
        console.log('❌ Failed tests:');
        results.tests
            .filter(t => !t.passed)
            .forEach(t => console.log(`   - ${t.name}: ${t.message}`));
    }

    console.log('═'.repeat(60));
}

// Run tests
main().catch(error => {
    console.error('');
    console.error('💥 Test execution failed:');
    console.error(error);
    console.error('');
    console.error('Stack trace:');
    console.error(error.stack);
    process.exit(1);
});
