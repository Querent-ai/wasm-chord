#!/usr/bin/env node
// Debug model creation with headless browser
const puppeteer = require('puppeteer');

async function debugModelCreation() {
    console.log('🔍 Debugging model creation...');
    
    let browser;
    try {
        browser = await puppeteer.launch({
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
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
        
        // Load the debug page
        await page.goto('http://localhost:8000/debug-model-creation.html', {
            waitUntil: 'networkidle0',
            timeout: 30000
        });
        
        // Wait for debug to complete
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Get the log content
        const logContent = await page.evaluate(() => {
            const logDiv = document.getElementById('log');
            return logDiv.textContent;
        });
        
        console.log('\n📋 Debug Results:');
        console.log(logContent);
        
    } catch (error) {
        console.error('❌ Debug failed:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

debugModelCreation().catch(console.error);
