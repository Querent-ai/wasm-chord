#!/usr/bin/env node
// Simple headless browser test for wasm-chord async API verification
const puppeteer = require('puppeteer');

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
        console.log('📄 Loading test page...');
        await page.goto('http://localhost:8000/test-async-verification.html', {
            waitUntil: 'networkidle0',
            timeout: 30000
        });
        
        console.log('✅ Test page loaded successfully');
        
        // Wait for WASM initialization
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Run basic API test
        console.log('🧪 Running basic API test...');
        await page.evaluate(() => {
            return window.testBasicAPI();
        });
        
        // Wait for test to complete
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Run async API test
        console.log('🧪 Running async API test...');
        await page.evaluate(() => {
            return window.testAsyncAPI();
        });
        
        // Wait for test to complete
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Run WebGPU test
        console.log('🧪 Running WebGPU test...');
        await page.evaluate(() => {
            return window.testWebGPU();
        });
        
        // Wait for test to complete
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Get test results
        const testResults = await page.evaluate(() => {
            const results = document.getElementById('test-results');
            const items = results.querySelectorAll('.test-item');
            const resultsArray = [];
            
            items.forEach(item => {
                const status = item.classList.contains('passing') ? 'PASS' : 
                             item.classList.contains('failing') ? 'FAIL' : 'WARN';
                const text = item.textContent.trim();
                resultsArray.push({ status, text });
            });
            
            return resultsArray;
        });
        
        console.log('\n📊 Test Results:');
        testResults.forEach(result => {
            const icon = result.status === 'PASS' ? '✅' : 
                        result.status === 'FAIL' ? '❌' : '⚠️';
            console.log(`${icon} ${result.text}`);
        });
        
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
        
        // Count results
        const passCount = testResults.filter(r => r.status === 'PASS').length;
        const failCount = testResults.filter(r => r.status === 'FAIL').length;
        const warnCount = testResults.filter(r => r.status === 'WARN').length;
        
        console.log('\n📈 Summary:');
        console.log(`Total Tests: ${testResults.length}`);
        console.log(`Passed: ${passCount}`);
        console.log(`Failed: ${failCount}`);
        console.log(`Warnings: ${warnCount}`);
        console.log(`Success Rate: ${testResults.length > 0 ? Math.round((passCount / testResults.length) * 100) : 0}%`);
        
        if (failCount === 0) {
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
