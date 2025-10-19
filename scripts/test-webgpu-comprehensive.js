#!/usr/bin/env node
// Comprehensive WebGPU test comparing headless vs real browser
const puppeteer = require('puppeteer');

async function testWebGPUComprehensive() {
    console.log('🧪 Comprehensive WebGPU Test');
    console.log('============================');
    
    let browser;
    try {
        // Test 1: Headless browser
        console.log('\n🔍 Testing in HEADLESS browser...');
        browser = await puppeteer.launch({
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--enable-features=WebGPU'
            ]
        });
        
        const headlessPage = await browser.newPage();
        
        headlessPage.on('console', msg => {
            console.log(`[HEADLESS ${msg.type().toUpperCase()}] ${msg.text()}`);
        });
        
        await headlessPage.goto('http://localhost:8000/webgpu-test.html', {
            waitUntil: 'networkidle0',
            timeout: 30000
        });
        
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Get headless results
        const headlessResults = await headlessPage.evaluate(() => {
            const results = document.getElementById('results');
            return results.textContent;
        });
        
        console.log('\n📋 Headless Results:');
        console.log(headlessResults);
        
        await browser.close();
        
        // Test 2: Non-headless browser (if possible)
        console.log('\n🔍 Testing in NON-HEADLESS browser...');
        browser = await puppeteer.launch({
            headless: false, // Try non-headless
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--enable-features=WebGPU',
                '--disable-gpu-sandbox'
            ]
        });
        
        const realPage = await browser.newPage();
        
        realPage.on('console', msg => {
            console.log(`[REAL ${msg.type().toUpperCase()}] ${msg.text()}`);
        });
        
        await realPage.goto('http://localhost:8000/webgpu-test.html', {
            waitUntil: 'networkidle0',
            timeout: 30000
        });
        
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Get real browser results
        const realResults = await realPage.evaluate(() => {
            const results = document.getElementById('results');
            return results.textContent;
        });
        
        console.log('\n📋 Real Browser Results:');
        console.log(realResults);
        
        // Test 3: Direct JavaScript evaluation
        console.log('\n🔍 Testing direct JavaScript evaluation...');
        const jsResults = await realPage.evaluate(() => {
            const results = [];
            
            // Test navigator.gpu
            results.push(`navigator.gpu exists: ${'gpu' in navigator}`);
            
            // Test WebGPU API
            if ('gpu' in navigator) {
                results.push(`navigator.gpu.requestAdapter exists: ${typeof navigator.gpu.requestAdapter === 'function'}`);
            }
            
            // Test wasm-chord detection
            return results;
        });
        
        console.log('📊 Direct JS Results:');
        jsResults.forEach(result => console.log(`  ${result}`));
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

testWebGPUComprehensive().catch(console.error);
