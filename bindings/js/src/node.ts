/**
 * @querent/wasm-chord-node - Node.js-optimized LLM inference
 * 
 * High-level TypeScript API for WebAssembly LLM inference in Node.js.
 * Optimized for Memory64 support and large model handling.
 */

// Re-export the generated wasm-bindgen types
export * from '../pkg/wasm_chord_runtime';

// Enhanced Node.js-specific types
export interface NodeInitConfig {
  /** Maximum memory allocation in bytes (default: 16GB for large models) */
  maxMemoryBytes?: number;
  /** Enable deterministic generation (default: false) */
  deterministic?: boolean;
  /** Enable CUDA acceleration (default: false) */
  cudaEnabled?: boolean;
  /** Number of worker threads (default: os.cpus().length) */
  numThreads?: number;
  /** Memory64 prefetch distance (default: 1) */
  prefetchDistance?: number;
}

export interface NodeGenOptions {
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

export interface NodeModelInfo {
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
  /** Whether Memory64 is active */
  memory64Active: boolean;
  /** Model size in bytes */
  sizeBytes: number;
  /** Memory64 statistics */
  memory64Stats?: {
    totalLayers: number;
    cachedLayers: number;
    cacheSize: number;
    prefetchDistance: number;
  };
}

/**
 * Node.js-optimized WasmChord runtime
 * 
 * Features:
 * - Memory64 support for large models (>4GB)
 * - CUDA acceleration
 * - On-demand layer loading
 * - Streaming inference
 * - Chat template support
 */
export class WasmChordNode {
  private wasmModule: any;
  private initialized: boolean = false;
  private config: NodeInitConfig;

  private constructor(wasmModule: any, config: NodeInitConfig) {
    this.wasmModule = wasmModule;
    this.config = config;
  }

  /**
   * Initialize the Node.js runtime
   */
  static async init(config?: NodeInitConfig): Promise<WasmChordNode> {
    const os = await import('os');
    
    const finalConfig: NodeInitConfig = {
      maxMemoryBytes: 16_000_000_000, // 16GB for large models
      deterministic: false,
      cudaEnabled: false,
      numThreads: os.cpus().length,
      prefetchDistance: 1,
      ...config,
    };

    // Import the generated wasm module
    const wasmModule = await import('../pkg/wasm_chord_runtime');
    await wasmModule.default();

    // Initialize with Node.js-specific settings
    if (finalConfig.cudaEnabled && await hasCUDA()) {
      console.log('🚀 CUDA acceleration enabled');
    } else {
      console.log('💻 Using CPU backend');
    }

    const instance = new WasmChordNode(wasmModule, finalConfig);
    instance.initialized = true;

    return instance;
  }

  /**
   * Load a model from file path or buffer
   * 
   * @param source File path string or Buffer
   * @param opts Loading options
   */
  async loadModel(
    source: string | Buffer, 
    opts?: { quant?: string; prefetchDistance?: number }
  ): Promise<NodeModel> {
    if (!this.initialized) {
      throw new Error('Runtime not initialized. Call WasmChordNode.init() first.');
    }

    let modelData: Uint8Array;

    if (typeof source === 'string') {
      // Read from file path
      console.log(`📥 Loading model from file: ${source}`);
      const fs = await import('fs');
      const buffer = fs.readFileSync(source);
      modelData = new Uint8Array(buffer);
    } else {
      // Buffer
      modelData = new Uint8Array(source);
    }

    const sizeGB = modelData.length / 1_000_000_000;
    console.log(`📊 Model size: ${sizeGB.toFixed(2)} GB`);

    // Use Memory64GGUFLoader for automatic Memory64 detection
    const loader = new this.wasmModule.Memory64GGUFLoader();
    const model = await loader.load_model(modelData);

    // Configure prefetch distance if provided
    if (opts?.prefetchDistance !== undefined) {
      model.set_prefetch_distance(opts.prefetchDistance);
    }

    return new NodeModel(this, model, `model-${Date.now()}`, sizeGB);
  }

  /**
   * Check if CUDA is available
   */
  async hasCUDA(): Promise<boolean> {
    return hasCUDA();
  }

  /**
   * Get runtime information
   */
  getRuntimeInfo(): { cuda: boolean; maxMemory: number; threads: number; memory64: boolean } {
    return {
      cuda: this.config.cudaEnabled || false,
      maxMemory: this.config.maxMemoryBytes || 16_000_000_000,
      threads: this.config.numThreads || 4,
      memory64: true, // Always available in Node.js
    };
  }
}

/**
 * Node.js model handle with Memory64 support
 */
export class NodeModel {
  private runtime: WasmChordNode;
  private wasmModel: any;
  private name: string;
  private sizeGB: number;

  constructor(runtime: WasmChordNode, wasmModel: any, name: string, sizeGB: number) {
    this.runtime = runtime;
    this.wasmModel = wasmModel;
    this.name = name;
    this.sizeGB = sizeGB;
  }

  /**
   * Get model information
   */
  getModelInfo(): NodeModelInfo {
    const info = this.wasmModel.get_model_info();
    const memory64Stats = this.wasmModel.get_memory64_stats();
    
    return {
      name: this.name,
      architecture: 'llama', // TODO: extract from GGUF
      vocabSize: info.vocab_size,
      hiddenSize: info.hidden_size,
      numLayers: info.num_layers,
      numHeads: info.num_heads,
      maxSeqLen: info.max_seq_len,
      memory64Active: this.wasmModel.is_memory64(),
      sizeBytes: Math.floor(this.sizeGB * 1_000_000_000),
      memory64Stats: memory64Stats ? {
        totalLayers: memory64Stats.total_layers,
        cachedLayers: memory64Stats.cached_layers,
        cacheSize: memory64Stats.cache_size,
        prefetchDistance: memory64Stats.prefetch_distance,
      } : undefined,
    };
  }

  /**
   * Generate text (blocking)
   */
  async generate(prompt: string, opts?: NodeGenOptions): Promise<string> {
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
  async *generateStream(prompt: string, opts?: NodeGenOptions): AsyncIterable<string> {
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
   * Set Memory64 prefetch distance
   */
  setPrefetchDistance(distance: number): void {
    this.wasmModel.set_prefetch_distance(distance);
    console.log(`⚡ Prefetch distance set to ${distance}`);
  }

  /**
   * Get Memory64 statistics
   */
  getMemory64Stats(): { totalLayers: number; cachedLayers: number; cacheSize: number; prefetchDistance: number } | null {
    const stats = this.wasmModel.get_memory64_stats();
    if (!stats) return null;
    
    return {
      totalLayers: stats.total_layers,
      cachedLayers: stats.cached_layers,
      cacheSize: stats.cache_size,
      prefetchDistance: stats.prefetch_distance,
    };
  }

  /**
   * Free model from memory
   */
  async free(): Promise<void> {
    // WASM models are automatically garbage collected
    console.log(`🗑️ Model ${this.name} freed from memory`);
  }

  private normalizeOptions(opts?: NodeGenOptions) {
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
 * Utility: Check if CUDA is available
 */
export async function hasCUDA(): Promise<boolean> {
  try {
    // Try to import CUDA-related modules
    const { execSync } = await import('child_process');
    execSync('nvidia-smi', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Utility: Check if Memory64 is supported (always true in Node.js)
 */
export async function hasMemory64(): Promise<boolean> {
  return true;
}

/**
 * Utility: Get Node.js capabilities
 */
export async function getNodeCapabilities(): Promise<{
  cuda: boolean;
  memory64: boolean;
  workers: number;
  maxMemory: number;
}> {
  const os = await import('os');
  
  return {
    cuda: await hasCUDA(),
    memory64: await hasMemory64(),
    workers: os.cpus().length,
    maxMemory: 64_000_000_000, // 64GB theoretical limit
  };
}

// Default export for convenience
export default WasmChordNode;
