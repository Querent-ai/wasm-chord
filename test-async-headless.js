#!/usr/bin/env node
// Headless browser test for wasm-chord async API
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function testAsyncAPI() {
    console.log('🧪 Starting headless browser test for wasm-chord async API...');
    
    let browser;
    try {
        // Launch headless browser
        browser = await puppeteer.launch({
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--enable-features=WebGPU'
            ]
        });
        
        const page = await browser.newPage();
        
        // Set up console logging
        page.on('console', msg => {
            const type = msg.type();
            const text = msg.text();
            console.log(`[${type.toUpperCase()}] ${text}`);
        });
        
        // Set up error handling
        page.on('pageerror', error => {
            console.error(`[PAGE ERROR] ${error.message}`);
        });
        
        // Load the test page
        const testPagePath = path.resolve(__dirname, 'test-async-api.html');
        console.log(`📄 Loading test page: ${testPagePath}`);
        
        await page.goto(`file://${testPagePath}`, {
            waitUntil: 'networkidle0',
            timeout: 30000
        });
        
        console.log('✅ Test page loaded successfully');
        
        // Wait for WASM module to initialize
        await page.waitForFunction(() => {
            return window.wasmModule !== null;
        }, { timeout: 10000 }).catch(() => {
            console.log('⚠️ WASM module initialization timeout, continuing...');
        });
        
        console.log('✅ WASM module initialized');
        
        // Run all tests
        console.log('🧪 Running all async API tests...');
        await page.evaluate(() => {
            return window.runAllTests();
        });
        
        // Wait for tests to complete
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Get test results
        const testResults = await page.evaluate(() => {
            return {
                total: window.testResults.total,
                passed: window.testResults.passed,
                failed: window.testResults.failed,
                warnings: window.testResults.warnings
            };
        });
        
        console.log('\n📊 Test Results:');
        console.log(`Total Tests: ${testResults.total}`);
        console.log(`Passed: ${testResults.passed}`);
        console.log(`Failed: ${testResults.failed}`);
        console.log(`Warnings: ${testResults.warnings}`);
        console.log(`Success Rate: ${testResults.total > 0 ? Math.round((testResults.passed / testResults.total) * 100) : 0}%`);
        
        // Test specific async functions
        console.log('\n🔍 Testing specific async functions...');
        
        // Test async GPU initialization
        try {
            await page.evaluate(() => {
                return window.testAsyncGPU();
            });
            console.log('✅ Async GPU test completed');
        } catch (error) {
            console.error('❌ Async GPU test failed:', error.message);
        }
        
        // Test async generation
        try {
            await page.evaluate(() => {
                return window.testAsyncGeneration();
            });
            console.log('✅ Async generation test completed');
        } catch (error) {
            console.error('❌ Async generation test failed:', error.message);
        }
        
        // Test streaming
        try {
            await page.evaluate(() => {
                return window.testStreaming();
            });
            console.log('✅ Streaming test completed');
        } catch (error) {
            console.error('❌ Streaming test failed:', error.message);
        }
        
        // Get browser information
        const browserInfo = await page.evaluate(() => {
            return {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                webGPU: 'gpu' in navigator,
                webAssembly: typeof WebAssembly === 'object'
            };
        });
        
        console.log('\n🌐 Browser Information:');
        console.log(`User Agent: ${browserInfo.userAgent}`);
        console.log(`Platform: ${browserInfo.platform}`);
        console.log(`WebGPU Support: ${browserInfo.webGPU ? '✅' : '❌'}`);
        console.log(`WebAssembly Support: ${browserInfo.webAssembly ? '✅' : '❌'}`);
        
        // Test WebGPU availability
        const webgpuAvailable = await page.evaluate(() => {
            return window.WasmModel.is_gpu_available();
        });
        console.log(`wasm-chord WebGPU Available: ${webgpuAvailable ? '✅' : '❌'}`);
        
        // Generate test report
        const report = {
            timestamp: new Date().toISOString(),
            browser: browserInfo,
            testResults: testResults,
            webgpuAvailable: webgpuAvailable,
            status: testResults.failed === 0 ? 'PASS' : 'FAIL'
        };
        
        fs.writeFileSync('async-api-test-report.json', JSON.stringify(report, null, 2));
        console.log('\n📄 Test report saved to: async-api-test-report.json');
        
        if (testResults.failed === 0) {
            console.log('\n🎉 All tests passed! Async API is working correctly.');
        } else {
            console.log('\n⚠️ Some tests failed. Check the logs above for details.');
        }
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        process.exit(1);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

// Run the test
testAsyncAPI().catch(console.error);
