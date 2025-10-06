// WASM-Chord Web Demo Application

import init, { WasmModel, format_chat, version } from './pkg/wasm_chord_runtime.js';

let model = null;
let conversationHistory = [];

// UI Elements
const statusEl = document.getElementById('status');
const modelInfoEl = document.getElementById('model-info');
const messagesEl = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');
const modelFile = document.getElementById('model-file');
const loadBtn = document.getElementById('load-btn');
const loadProgress = document.getElementById('load-progress');
const progressText = document.getElementById('progress-text');
const versionEl = document.getElementById('version');

// Config sliders
const maxTokensSlider = document.getElementById('max-tokens');
const temperatureSlider = document.getElementById('temperature');
const maxTokensVal = document.getElementById('max-tokens-val');
const temperatureVal = document.getElementById('temperature-val');

// Initialize WASM
async function initWasm() {
    try {
        await init();
        versionEl.textContent = version();
        statusEl.textContent = 'WASM loaded - Ready to load model';
        console.log('WASM initialized, version:', version());
    } catch (error) {
        console.error('Failed to initialize WASM:', error);
        statusEl.textContent = 'Error loading WASM';
    }
}

// Load model from file
async function loadModel() {
    const file = modelFile.files[0];
    if (!file) {
        statusEl.textContent = 'Please select a model file';
        statusEl.style.color = '#f44336';
        return;
    }

    // Validate file extension
    if (!file.name.endsWith('.gguf')) {
        statusEl.textContent = 'Invalid file - must be a .gguf model file';
        statusEl.style.color = '#f44336';
        return;
    }

    // Check file size (warn if > 2GB)
    const fileSizeMB = file.size / 1024 / 1024;
    if (fileSizeMB > 2048) {
        if (!confirm(`Large model file (${fileSizeMB.toFixed(0)} MB). This may cause out-of-memory errors. Continue?`)) {
            return;
        }
    }

    try {
        statusEl.textContent = 'Loading model...';
        statusEl.style.color = '';
        loadProgress.style.display = 'block';
        progressText.textContent = 'Reading file...';
        loadBtn.disabled = true;

        // Read file as ArrayBuffer
        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);

        progressText.textContent = `Initializing model (${fileSizeMB.toFixed(1)} MB)...`;

        // Load model with timeout detection
        const loadTimeout = setTimeout(() => {
            progressText.textContent = 'This is taking longer than expected...';
        }, 5000);

        model = new WasmModel(bytes);
        clearTimeout(loadTimeout);

        // Set default config
        updateModelConfig();

        statusEl.textContent = 'Model loaded ✓';
        statusEl.style.color = '#4caf50';
        modelInfoEl.textContent = `${file.name} (${fileSizeMB.toFixed(1)} MB)`;
        loadProgress.style.display = 'none';

        // Enable chat
        userInput.disabled = false;
        sendBtn.disabled = false;
        clearBtn.disabled = false;

        addMessage('assistant', 'Hello! I\'m ready to chat. How can I help you today?');

    } catch (error) {
        console.error('Failed to load model:', error);

        // Provide user-friendly error messages
        let errorMsg = 'Failed to load model';
        if (error.message.includes('memory')) {
            errorMsg = 'Out of memory - try a smaller model or close other tabs';
        } else if (error.message.includes('parse') || error.message.includes('invalid')) {
            errorMsg = 'Invalid or corrupted model file';
        } else if (error.message) {
            errorMsg = `Error: ${error.message}`;
        }

        statusEl.textContent = errorMsg;
        statusEl.style.color = '#f44336';
        progressText.textContent = errorMsg;

        loadProgress.style.display = 'none';
        loadBtn.disabled = false;
    }
}

// Update model configuration from sliders
function updateModelConfig() {
    if (!model) return;

    const maxTokens = parseInt(maxTokensSlider.value);
    const temperature = parseFloat(temperatureSlider.value);
    const topP = 0.95;
    const topK = 40;
    const repetitionPenalty = 1.1;

    model.set_config(maxTokens, temperature, topP, topK, repetitionPenalty);
}

// Add message to chat
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = content;
    
    messageDiv.appendChild(bubble);
    messagesEl.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Add typing indicator
function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typing-indicator';
    
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    
    messageDiv.appendChild(indicator);
    messagesEl.appendChild(messageDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Send message and generate response
async function sendMessage() {
    if (!model || !userInput.value.trim()) return;

    const userMessage = userInput.value.trim();
    userInput.value = '';
    
    // Add user message to UI
    addMessage('user', userMessage);
    conversationHistory.push({ role: 'user', content: userMessage });

    // Disable input during generation
    userInput.disabled = true;
    sendBtn.disabled = true;
    addTypingIndicator();

    try {
        // Format prompt with chat template
        const systemPrompt = "You are a helpful, friendly AI assistant. Keep responses concise and relevant.";
        const prompt = format_chat(systemPrompt, userMessage, 'chatml');

        // Generate with streaming
        let assistantResponse = '';
        const startTime = Date.now();
        let tokenCount = 0;

        // Set generation timeout (30 seconds)
        const generationTimeout = setTimeout(() => {
            console.warn('Generation is taking longer than expected');
        }, 30000);

        await model.generate_stream(prompt, (tokenText) => {
            assistantResponse += tokenText;
            tokenCount++;

            // Update the last message
            removeTypingIndicator();
            const lastMsg = messagesEl.lastElementChild;
            if (lastMsg && lastMsg.classList.contains('assistant')) {
                lastMsg.querySelector('.message-bubble').textContent = assistantResponse;
            } else {
                addMessage('assistant', assistantResponse);
            }

            messagesEl.scrollTop = messagesEl.scrollHeight;
            return true; // Continue generation
        });

        clearTimeout(generationTimeout);

        const duration = (Date.now() - startTime) / 1000;
        const tokensPerSec = tokenCount / duration;
        console.log(`Generation: ${tokenCount} tokens in ${duration.toFixed(1)}s (${tokensPerSec.toFixed(1)} tokens/s)`);

        // Add to history
        conversationHistory.push({ role: 'assistant', content: assistantResponse });

        // Warn if generation was very slow
        if (tokensPerSec < 0.5) {
            console.warn('Generation is slow. Consider using a smaller model or enabling GPU acceleration.');
        }

    } catch (error) {
        console.error('Generation failed:', error);
        removeTypingIndicator();

        // Provide user-friendly error messages
        let errorMsg = 'Sorry, an error occurred during generation.';
        if (error.message.includes('memory')) {
            errorMsg = 'Out of memory during generation. Try reducing max tokens or using a smaller model.';
        } else if (error.message.includes('timeout')) {
            errorMsg = 'Generation timed out. Try again or reduce max tokens.';
        } else if (error.message) {
            errorMsg = `Generation error: ${error.message}`;
        }

        addMessage('assistant', errorMsg);
        statusEl.textContent = 'Generation failed';
        statusEl.style.color = '#f44336';
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// Clear conversation
function clearConversation() {
    if (!confirm('Clear conversation history?')) return;
    
    messagesEl.innerHTML = '';
    conversationHistory = [];
    addMessage('assistant', 'Conversation cleared. How can I help you?');
}

// Event listeners
loadBtn.addEventListener('click', loadModel);
sendBtn.addEventListener('click', sendMessage);
clearBtn.addEventListener('click', clearConversation);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Update slider values
maxTokensSlider.addEventListener('input', (e) => {
    maxTokensVal.textContent = e.target.value;
    updateModelConfig();
});

temperatureSlider.addEventListener('input', (e) => {
    temperatureVal.textContent = parseFloat(e.target.value).toFixed(1);
    updateModelConfig();
});

// Initialize on load
window.addEventListener('load', initWasm);
