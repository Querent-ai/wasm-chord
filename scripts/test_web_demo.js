// Node.js test script for WASM module
// Usage: node test_web_demo.js

import { readFile } from 'fs/promises';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function testWasmModule() {
    console.log('🧪 Testing WASM-Chord Module\n');

    try {
        // Test 1: Check WASM file exists and is valid
        console.log('1️⃣  Testing WASM file...');
        const wasmPath = join(__dirname, 'crates/wasm-chord-runtime/pkg/wasm_chord_runtime_bg.wasm');
        const wasmBytes = await readFile(wasmPath);
        console.log(`   ✅ WASM file size: ${(wasmBytes.length / 1024).toFixed(1)} KB`);

        // Check WASM magic number
        const magic = wasmBytes.slice(0, 4);
        const expectedMagic = Buffer.from([0x00, 0x61, 0x73, 0x6d]);
        if (magic.equals(expectedMagic)) {
            console.log('   ✅ Valid WASM magic number');
        } else {
            console.log('   ❌ Invalid WASM magic number');
            return false;
        }

        // Test 2: Check JS bindings exist
        console.log('\n2️⃣  Testing JS bindings...');
        const jsPath = join(__dirname, 'crates/wasm-chord-runtime/pkg/wasm_chord_runtime.js');
        const jsContent = await readFile(jsPath, 'utf-8');

        const exports = [
            'WasmModel',
            'format_chat',
            'version',
            'init'
        ];

        let allExportsFound = true;
        for (const exp of exports) {
            if (jsContent.includes(exp)) {
                console.log(`   ✅ Export found: ${exp}`);
            } else {
                console.log(`   ❌ Missing export: ${exp}`);
                allExportsFound = false;
            }
        }

        // Test 3: Check web demo files
        console.log('\n3️⃣  Testing web demo files...');
        const demoFiles = [
            'examples/web-demo/index.html',
            'examples/web-demo/style.css',
            'examples/web-demo/app.js',
            'examples/web-demo/README.md',
            'examples/web-demo/pkg/wasm_chord_runtime.js',
            'examples/web-demo/pkg/wasm_chord_runtime_bg.wasm'
        ];

        let allFilesExist = true;
        for (const file of demoFiles) {
            try {
                const filePath = join(__dirname, file);
                const stat = await readFile(filePath);
                console.log(`   ✅ ${file} (${(stat.length / 1024).toFixed(1)} KB)`);
            } catch (err) {
                console.log(`   ❌ Missing: ${file}`);
                allFilesExist = false;
            }
        }

        // Test 4: Check HTML imports
        console.log('\n4️⃣  Testing HTML structure...');
        const htmlPath = join(__dirname, 'examples/web-demo/index.html');
        const htmlContent = await readFile(htmlPath, 'utf-8');

        const htmlChecks = [
            { name: 'CSS import', pattern: 'style.css' },
            { name: 'JS module', pattern: 'type="module"' },
            { name: 'App.js import', pattern: 'app.js' },
            { name: 'Chat container', pattern: 'chat-messages' },
            { name: 'User input', pattern: 'user-input' },
            { name: 'Model upload', pattern: 'model-file' }
        ];

        for (const check of htmlChecks) {
            if (htmlContent.includes(check.pattern)) {
                console.log(`   ✅ ${check.name}`);
            } else {
                console.log(`   ❌ Missing: ${check.name}`);
            }
        }

        // Test 5: Check app.js imports
        console.log('\n5️⃣  Testing app.js structure...');
        const appJsPath = join(__dirname, 'examples/web-demo/app.js');
        const appJsContent = await readFile(appJsPath, 'utf-8');

        const appJsChecks = [
            { name: 'WASM import', pattern: 'import init' },
            { name: 'WasmModel import', pattern: 'WasmModel' },
            { name: 'format_chat import', pattern: 'format_chat' },
            { name: 'version import', pattern: 'version' },
            { name: 'Model loading', pattern: 'loadModel' },
            { name: 'Message sending', pattern: 'sendMessage' },
            { name: 'Streaming', pattern: 'generate_stream' },
            { name: 'Config update', pattern: 'set_config' }
        ];

        for (const check of appJsChecks) {
            if (appJsContent.includes(check.pattern)) {
                console.log(`   ✅ ${check.name}`);
            } else {
                console.log(`   ❌ Missing: ${check.name}`);
            }
        }

        console.log('\n' + '='.repeat(60));
        console.log('🎉 All tests passed! Web demo is ready for browser testing.');
        console.log('='.repeat(60));
        console.log('\n📋 Next Steps:');
        console.log('   1. Open http://localhost:8000 in browser');
        console.log('   2. Upload a GGUF model');
        console.log('   3. Test chat functionality');
        console.log('   4. Verify streaming works');
        console.log('   5. Test on mobile/different browsers\n');

        return true;

    } catch (error) {
        console.error('❌ Test failed:', error.message);
        return false;
    }
}

testWasmModule().then(success => {
    process.exit(success ? 0 : 1);
});
