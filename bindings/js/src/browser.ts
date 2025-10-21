/**
 * @querent/wasm-chord-web - Browser-optimized LLM inference
 * 
 * High-level TypeScript API for WebAssembly LLM inference in browsers.
 * Optimized for WebGPU acceleration and browser memory constraints.
 */

// Re-export the generated wasm-bindgen types
export * from '../pkg/wasm_chord_runtime';

// Enhanced browser-specific types
export interface BrowserInitConfig {
  /** Maximum memory allocation in bytes (default: 2GB for browser safety) */
  maxMemoryBytes?: number;
  /** Enable deterministic generation (default: false) */
  deterministic?: boolean;
  /** Enable WebGPU acceleration (default: true) */
  webgpuEnabled?: boolean;
  /** Number of worker threads (default: navigator.hardwareConcurrency) */
  numThreads?: number;
}

export interface BrowserGenOptions {
  /** Maximum tokens to generate (default: 512) */
  maxTokens?: number;
  /** Sampling temperature 0.0-2.0 (default: 0.7) */
  temperature?: number;
  /** Top-k sampling (default: 40) */
  topK?: number;
  /** Top-p sampling (default: 0.9) */
  topP?: number;
  /** Random seed (default: 0) */
  seed?: number;
  /** Stop token IDs (default: []) */
  stopTokens?: number[];
  /** Use chat template (default: true) */
  useChatTemplate?: boolean;
}

export interface BrowserModelInfo {
  /** Model name */
  name: string;
  /** Model architecture */
  architecture: string;
  /** Vocabulary size */
  vocabSize: number;
  /** Hidden dimension size */
  hiddenSize: number;
  /** Number of transformer layers */
  numLayers: number;
  /** Number of attention heads */
  numHeads: number;
  /** Maximum sequence length */
  maxSeqLen: number;
  /** Whether this is a browser-compatible model */
  isBrowserModel: boolean;
  /** Model size in bytes */
  sizeBytes: number;
}

/**
 * Browser-optimized WasmChord runtime
 * 
 * Features:
 * - WebGPU acceleration
 * - Browser memory safety (<4GB models)
 * - Streaming inference
 * - Chat template support
 */
export class WasmChordBrowser {
  private wasmModule: any;
  private initialized: boolean = false;
  private config: BrowserInitConfig;

  private constructor(wasmModule: any, config: BrowserInitConfig) {
    this.wasmModule = wasmModule;
    this.config = config;
  }

  /**
   * Initialize the browser runtime
   */
  static async init(config?: BrowserInitConfig): Promise<WasmChordBrowser> {
    const finalConfig: BrowserInitConfig = {
      maxMemoryBytes: 2_000_000_000, // 2GB browser safety limit
      deterministic: false,
      webgpuEnabled: true,
      numThreads: navigator.hardwareConcurrency || 4,
      ...config,
    };

    // Import the generated wasm module
    const wasmModule = await import('../pkg/wasm_chord_runtime');
    await wasmModule.default();

    // Initialize with browser-specific settings
    if (finalConfig.webgpuEnabled && await hasWebGPU()) {
      console.log('🚀 WebGPU acceleration enabled');
    } else {
      console.log('⚠️ WebGPU not available, using CPU fallback');
    }

    const instance = new WasmChordBrowser(wasmModule, finalConfig);
    instance.initialized = true;

    return instance;
  }

  /**
   * Load a model from various sources
   * 
   * @param source URL string, File object, or ArrayBuffer
   * @param opts Loading options
   */
  async loadModel(
    source: string | File | ArrayBuffer, 
    opts?: { quant?: string }
  ): Promise<BrowserModel> {
    if (!this.initialized) {
      throw new Error('Runtime not initialized. Call WasmChordBrowser.init() first.');
    }

    let modelData: Uint8Array;

    if (typeof source === 'string') {
      // Fetch from URL
      console.log(`📥 Loading model from URL: ${source}`);
      const response = await fetch(source);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
      }
      modelData = new Uint8Array(await response.arrayBuffer());
    } else if (source instanceof File) {
      // Read from File
      console.log(`📥 Loading model from file: ${source.name}`);
      modelData = new Uint8Array(await source.arrayBuffer());
    } else {
      // ArrayBuffer
      modelData = new Uint8Array(source);
    }

    // Check browser memory constraints
    const sizeGB = modelData.length / 1_000_000_000;
    if (sizeGB > 3.5) {
      throw new Error(
        `Model too large (${sizeGB.toFixed(2)} GB) for browser. ` +
        `Browser limit is ~4GB. Use @querent/wasm-chord-node for larger models.`
      );
    }

    console.log(`📊 Model size: ${sizeGB.toFixed(2)} GB`);

    // Create WasmModel instance
    const wasmModel = new this.wasmModule.WasmModel(modelData);
    
    return new BrowserModel(this, wasmModel, `model-${Date.now()}`);
  }

  /**
   * Check if WebGPU is available
   */
  async hasWebGPU(): Promise<boolean> {
    return hasWebGPU();
  }

  /**
   * Get runtime information
   */
  getRuntimeInfo(): { webgpu: boolean; maxMemory: number; threads: number } {
    return {
      webgpu: this.config.webgpuEnabled || false,
      maxMemory: this.config.maxMemoryBytes || 2_000_000_000,
      threads: this.config.numThreads || 4,
    };
  }
}

/**
 * Browser model handle with streaming support
 */
export class BrowserModel {
  private runtime: WasmChordBrowser;
  private wasmModel: any;
  private name: string;

  constructor(runtime: WasmChordBrowser, wasmModel: any, name: string) {
    this.runtime = runtime;
    this.wasmModel = wasmModel;
    this.name = name;
  }

  /**
   * Get model information
   */
  getModelInfo(): BrowserModelInfo {
    const info = this.wasmModel.get_model_info();
    return {
      name: this.name,
      architecture: 'llama', // TODO: extract from GGUF
      vocabSize: info.vocab_size,
      hiddenSize: info.hidden_size,
      numLayers: info.num_layers,
      numHeads: info.num_heads,
      maxSeqLen: info.max_seq_len,
      isBrowserModel: info.is_browser_model,
      sizeBytes: 0, // TODO: extract from GGUF
    };
  }

  /**
   * Generate text (blocking)
   */
  async generate(prompt: string, opts?: BrowserGenOptions): Promise<string> {
    const options = this.normalizeOptions(opts);
    
    console.log(`🎯 Generating with prompt: "${prompt.substring(0, 50)}..."`);
    
    // Use the async token stream and collect all tokens
    const tokens: string[] = [];
    for await (const token of this.generateStream(prompt, options)) {
      tokens.push(token);
    }
    
    return tokens.join('');
  }

  /**
   * Generate text (streaming)
   */
  async *generateStream(prompt: string, opts?: BrowserGenOptions): AsyncIterable<string> {
    const options = this.normalizeOptions(opts);
    
    console.log(`🎯 Streaming generation with prompt: "${prompt.substring(0, 50)}..."`);
    
    // Create async token stream
    const stream = this.wasmModel.generate_async(prompt, options.maxTokens);
    
    // Yield tokens as they arrive
    for await (const token of stream) {
      yield token;
    }
  }

  /**
   * Free model from memory
   */
  async free(): Promise<void> {
    // WASM models are automatically garbage collected
    console.log(`🗑️ Model ${this.name} freed from memory`);
  }

  private normalizeOptions(opts?: BrowserGenOptions) {
    return {
      maxTokens: opts?.maxTokens ?? 512,
      temperature: opts?.temperature ?? 0.7,
      topK: opts?.topK ?? 40,
      topP: opts?.topP ?? 0.9,
      seed: opts?.seed ?? 0,
      stopTokens: opts?.stopTokens ?? [],
      useChatTemplate: opts?.useChatTemplate ?? true,
    };
  }
}

/**
 * Utility: Check if WebGPU is available
 */
export async function hasWebGPU(): Promise<boolean> {
  if (typeof navigator === 'undefined') return false;
  
  try {
    return 'gpu' in navigator && await navigator.gpu.requestAdapter() !== null;
  } catch {
    return false;
  }
}

/**
 * Utility: Check if Memory64 is supported (browser limitation)
 */
export async function hasMemory64(): Promise<boolean> {
  // Memory64 is not supported in browsers yet
  return false;
}

/**
 * Utility: Get browser capabilities
 */
export async function getBrowserCapabilities(): Promise<{
  webgpu: boolean;
  memory64: boolean;
  workers: number;
  maxMemory: number;
}> {
  return {
    webgpu: await hasWebGPU(),
    memory64: await hasMemory64(),
    workers: navigator.hardwareConcurrency || 4,
    maxMemory: 4_000_000_000, // ~4GB browser limit
  };
}

// Default export for convenience
export default WasmChordBrowser;
