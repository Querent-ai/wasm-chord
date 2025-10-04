/**
 * wasm-chord TypeScript/JavaScript bindings
 *
 * High-level API for WebAssembly LLM inference runtime.
 */

export interface InitConfig {
  maxMemoryBytes?: number;
  deterministic?: boolean;
  gpuEnabled?: boolean;
  numThreads?: number;
}

export interface GenOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  seed?: number;
  stopTokens?: number[];
}

export interface ModelInfo {
  name: string;
  architecture: string;
  vocabSize?: number;
  tensorCount: number;
}

/**
 * Main WasmChord runtime class
 */
export class WasmChord {
  private wasmModule: any; // Type will be replaced with actual wasm-bindgen types
  private initialized: boolean = false;

  private constructor(wasmModule: any) {
    this.wasmModule = wasmModule;
  }

  /**
   * Initialize wasm-chord runtime
   */
  static async init(config?: InitConfig): Promise<WasmChord> {
    // In production, this would load the actual wasm module
    // For now, this is a placeholder structure

    // const wasmModule = await import('../pkg/wasm_chord_runtime');
    // await wasmModule.default(); // Initialize wasm

    const instance = new WasmChord(null);
    instance.initialized = true;

    return instance;
  }

  /**
   * Load a model from URL, File, or ReadableStream
   */
  async loadModel(source: string | File | ReadableStream, opts?: { quant?: string }): Promise<Model> {
    if (!this.initialized) {
      throw new Error('Runtime not initialized. Call WasmChord.init() first.');
    }

    let modelData: Uint8Array;

    if (typeof source === 'string') {
      // Fetch from URL
      const response = await fetch(source);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
      }
      modelData = new Uint8Array(await response.arrayBuffer());
    } else if (source instanceof File) {
      // Read from File
      modelData = new Uint8Array(await source.arrayBuffer());
    } else {
      // Stream (future implementation)
      throw new Error('ReadableStream loading not yet implemented');
    }

    // Call C ABI to load model
    // const modelHandle = this.wasmModule.wasmchord_load_model(...);

    return new Model(this, 1, 'loaded-model'); // Placeholder
  }

  /**
   * List loaded models
   */
  async listModels(): Promise<ModelInfo[]> {
    // Placeholder
    return [];
  }

  /**
   * Delete a model from memory
   */
  async deleteModel(id: string): Promise<void> {
    // Placeholder
  }
}

/**
 * Loaded model handle
 */
export class Model {
  private runtime: WasmChord;
  private handle: number;
  private name: string;

  constructor(runtime: WasmChord, handle: number, name: string) {
    this.runtime = runtime;
    this.handle = handle;
    this.name = name;
  }

  /**
   * Blocking inference
   */
  async infer(prompt: string, opts?: GenOptions): Promise<string> {
    const options = this.normalizeOptions(opts);

    // Call C ABI
    // const result = wasmchord_infer(...);

    return "Placeholder response"; // Placeholder
  }

  /**
   * Streaming inference
   */
  async *inferStream(prompt: string, opts?: GenOptions): AsyncIterable<string> {
    const options = this.normalizeOptions(opts);

    // In real implementation:
    // const streamHandle = wasmchord_infer(...);
    // while (true) {
    //   const token = wasmchord_next_token(streamHandle);
    //   if (!token) break;
    //   yield token;
    // }

    // Placeholder
    yield "token1";
    yield "token2";
  }

  /**
   * Generate embeddings (future)
   */
  async embed(text: string): Promise<number[]> {
    throw new Error('Embeddings not yet implemented');
  }

  /**
   * Free model from memory
   */
  async free(): Promise<void> {
    // Call C ABI to free model
    // wasmchord_free_model(this.handle);
  }

  private normalizeOptions(opts?: GenOptions) {
    return {
      maxTokens: opts?.maxTokens ?? 512,
      temperature: opts?.temperature ?? 0.7,
      topK: opts?.topK ?? 40,
      topP: opts?.topP ?? 0.9,
      seed: opts?.seed ?? 0,
      stopTokens: opts?.stopTokens ?? [],
    };
  }
}

/**
 * Utility: check if WebGPU is available
 */
export async function hasWebGPU(): Promise<boolean> {
  return 'gpu' in navigator;
}

/**
 * Utility: check if memory64 is supported
 */
export async function hasMemory64(): Promise<boolean> {
  try {
    // Try to create a memory64 instance
    new WebAssembly.Memory({ initial: 1, maximum: 1, address: 'i64' } as any);
    return true;
  } catch {
    return false;
  }
}
