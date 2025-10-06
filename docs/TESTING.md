# Web Demo Testing Guide

## Current Status

✅ **WASM Module Built**: 275KB compiled module ready
✅ **Web Server Running**: http://localhost:8000
✅ **WASM Files Accessible**: All required files serving correctly
✅ **Test Models Available**: TinyLlama models in /home/puneet/wasm-chord/models/

## Quick Test Steps

### 1. Test WASM Module Loading
Open in browser:
```
http://localhost:8000/test.html
```

Expected result: Green checkmark with version number

### 2. Test Full Chat Interface
Open in browser:
```
http://localhost:8000/
```

### 3. Load Model and Chat

1. **Click "Choose File"** and select one of:
   - `/home/puneet/wasm-chord/models/tinyllama-q8.gguf`
   - `/home/puneet/wasm-chord/models/tinyllama-q4km.gguf`
   - `/home/puneet/wasm-chord/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf`

2. **Click "Load Model"**
   - Wait 10-30 seconds for loading
   - Status should show "Model loaded ✓"

3. **Type a message**: "Hello, how are you?"

4. **Press Send**
   - Watch tokens stream in real-time
   - Should see typing indicator then streaming response

## Expected Performance

- **Model Loading**: 10-30 seconds (depending on size)
- **Token Generation**: ~3.5 seconds per token (CPU only)
- **Memory Usage**: ~1.5GB for TinyLlama Q8

## Troubleshooting

### WASM won't load
Check browser console (F12) for errors:
- CORS issues? Make sure serving from http-server
- Module not found? Check pkg/ directory exists

### Model won't load
- Check file is valid GGUF format
- Ensure enough RAM available
- Try smaller model (Q4_0 instead of Q8)

### Generation is slow
- Normal! CPU inference takes 3-5s per token
- Try reducing max tokens (slider)
- Or use smaller model

## Browser Console Commands

Open DevTools (F12) and try:
```javascript
// Check WASM loaded
window.wasm_module

// Check model status
window.model
```

## Next Steps After Testing

1. **Take Screenshots**: Show loading, chat interface, streaming
2. **Record Demo Video**: Show full workflow
3. **Document Issues**: Note any bugs or UX improvements
4. **Deploy to GitHub Pages**: Share publicly

## Stopping the Server

```bash
# Find and kill the Python server
pkill -f "python3 -m http.server 8000"
```

## Rebuilding After Changes

If you modify Rust code:
```bash
cd ../../crates/wasm-chord-runtime
wasm-pack build --target web --out-dir pkg
cp -r pkg ../../examples/web-demo/
```

If you modify HTML/CSS/JS:
- Just refresh browser (no rebuild needed)
