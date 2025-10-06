# WASM-Chord Web Demo

Browser-based LLM chat with real-time streaming - 100% local inference!

## Quick Start

### 1. Build WASM Module
```bash
cd ../../crates/wasm-chord-runtime
wasm-pack build --target web --out-dir pkg
```

### 2. Get a Model
Download TinyLlama 1.1B Q8:
```bash
# Using wget
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf

# Or visit: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
```

### 3. Serve the Demo
```bash
# Using Python
python3 -m http.server 8000

# Or using Node.js
npx http-server -p 8000

# Or any other static file server
```

### 4. Open in Browser
Navigate to: `http://localhost:8000`

## Usage

1. **Load Model**: Click "Load Model" and select your `.gguf` file
2. **Wait**: Model loading takes 10-30 seconds depending on size
3. **Chat**: Type your message and press Enter or click Send
4. **Watch**: Tokens stream in real-time!

## Features

- ✅ Real-time token streaming
- ✅ 100% local inference (no server!)
- ✅ Adjustable generation settings
- ✅ Chat history
- ✅ Mobile responsive
- ✅ Clean, modern UI

## Configuration

Adjust these settings before or during chat:

- **Max Tokens**: 10-200 (response length)
- **Temperature**: 0.0-1.0 (creativity vs consistency)

## Requirements

- **Browser**: Chrome 90+, Firefox 88+, Safari 14+
- **Memory**: 2-4 GB RAM
- **Model**: GGUF format (Q4_0, Q8_0)

## Supported Models

- TinyLlama 1.1B (recommended for demo)
- Phi-2 2.7B  
- Mistral 7B (requires more RAM)

## Troubleshooting

### Model won't load
- Check file is valid GGUF format
- Ensure enough RAM (model size + 1GB)
- Try smaller quantization (Q4_0 instead of Q8_0)

### Generation is slow
- Normal! CPU inference takes 3-5s per token
- Try smaller model or fewer max tokens
- WebGPU backend coming soon for speedup

### Page doesn't load
- Check WASM module is built (`pkg/` directory exists)
- Ensure serving from correct directory
- Check browser console for errors

## Performance

Current (CPU only):
- **Speed**: ~3.5s per token
- **Quality**: Good with proper prompts
- **Memory**: ~1.5GB for TinyLlama Q8

Future (with WebGPU):
- **Speed**: <0.5s per token (7-10x faster!)
- **Quality**: Same
- **Memory**: Similar

## Technical Details

### Architecture
```
Browser
  ├─ HTML/CSS/JS UI
  ├─ WASM Runtime
  │   ├─ Model Loading
  │   ├─ Tokenizer
  │   └─ Inference Engine
  └─ Streaming Callbacks
```

### API Usage
```javascript
import init, { WasmModel } from './pkg/wasm_chord_runtime.js';

// Initialize
await init();

// Load model
const bytes = new Uint8Array(await file.arrayBuffer());
const model = new WasmModel(bytes);

// Configure
model.set_config(50, 0.7, 0.95, 40, 1.1);

// Generate with streaming
model.generate_stream(prompt, (token_text) => {
    console.log(token_text);
    return true; // continue
});
```

## Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome  | 90+     | ✅ Full support |
| Firefox | 88+     | ✅ Full support |
| Safari  | 14+     | ✅ Full support |
| Edge    | 90+     | ✅ Full support |
| Mobile  | Latest  | ✅ Responsive |

## Privacy

- ✅ 100% local inference
- ✅ No data sent to servers
- ✅ Model runs in your browser
- ✅ No telemetry or tracking
- ✅ Works offline (after loading)

## Development

### File Structure
```
examples/web-demo/
├── index.html      # UI structure
├── style.css       # Styling
├── app.js          # Application logic
└── README.md       # This file
```

### Modify & Test
1. Edit files
2. Refresh browser (no rebuild needed for HTML/CSS/JS)
3. Rebuild WASM only if Rust code changes

## Deployment

### GitHub Pages
```bash
# 1. Copy WASM files
cp -r ../../crates/wasm-chord-runtime/pkg .

# 2. Commit to gh-pages branch
git checkout -b gh-pages
git add .
git commit -m "Deploy web demo"
git push origin gh-pages

# 3. Enable GitHub Pages in repo settings
```

### Other Hosts
- Netlify: Drop folder or connect repo
- Vercel: Deploy from Git
- Cloudflare Pages: Connect GitHub repo

## License

MIT or Apache-2.0 (same as main project)

## Links

- [Main Repo](https://github.com/querent-ai/wasm-chord)
- [Documentation](../../README.md)
- [Report Issues](https://github.com/querent-ai/wasm-chord/issues)

---

Built with ❤️ using [wasm-chord](https://github.com/querent-ai/wasm-chord)
