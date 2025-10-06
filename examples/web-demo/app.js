// WASM-Chord Web Demo Application

import init, { WasmModel, format_chat, version } from '../../crates/wasm-chord-runtime/pkg/wasm_chord_runtime.js';

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
        alert('Please select a model file');
        return;
    }

    try {
        statusEl.textContent = 'Loading model...';
        loadProgress.style.display = 'block';
        progressText.textContent = 'Reading file...';
        loadBtn.disabled = true;

        // Read file as ArrayBuffer
        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);

        progressText.textContent = 'Initializing model...';
        
        // Load model
        model = new WasmModel(bytes);
        
        // Set default config
        updateModelConfig();

        statusEl.textContent = 'Model loaded ✓';
        modelInfoEl.textContent = `${file.name} (${(bytes.length / 1024 / 1024).toFixed(1)} MB)`;
        loadProgress.style.display = 'none';

        // Enable chat
        userInput.disabled = false;
        sendBtn.disabled = false;
        clearBtn.disabled = false;
        
        addMessage('assistant', 'Hello! I\'m ready to chat. How can I help you today?');

    } catch (error) {
        console.error('Failed to load model:', error);
        statusEl.textContent = 'Error loading model';
        alert('Failed to load model: ' + error.message);
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

        await model.generate_stream(prompt, (tokenText) => {
            assistantResponse += tokenText;
            
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

        const duration = (Date.now() - startTime) / 1000;
        console.log(`Generation took ${duration.toFixed(1)}s`);

        // Add to history
        conversationHistory.push({ role: 'assistant', content: assistantResponse });

    } catch (error) {
        console.error('Generation failed:', error);
        removeTypingIndicator();
        addMessage('assistant', 'Sorry, an error occurred during generation.');
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
