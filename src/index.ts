import { GeminiProvider } from "./providers/gemini-provider";


export interface LLMOptions {
  model?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxOutputTokens?: number;
  systemInstruction?: string;
}

export interface LLMResult {
  text: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  raw?: any;
}

export interface LLMChunk {
  text: string;
  isLast: boolean;
  raw?: any;
}

export interface ILLMProvider {
  generate(prompt: string, options?: LLMOptions): Promise<LLMResult>;
  generateStream(prompt: string, options?: LLMOptions): AsyncIterable<LLMChunk>;
  getName(): string;
}

export type ProviderType = 'gemini' | 'openai' | 'local' | 'openrouter';


export class LoadBalancerProvider implements ILLMProvider {
  private instances: ILLMProvider[];
  private currentIndex: number = 0;

  constructor(instances: ILLMProvider[]) {
    if (instances.length === 0) {
      throw new Error("LoadBalancerProvider requires at least one provider instance.");
    }
    this.instances = instances;
  }

  private getNextInstance(): ILLMProvider {
    const instance = this.instances[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.instances.length;
    return instance;
  }

  async generate(prompt: string, options?: LLMOptions): Promise<LLMResult> {
    return this.getNextInstance().generate(prompt, options);
  }

  async *generateStream(prompt: string, options?: LLMOptions): AsyncIterable<LLMChunk> {
    yield* this.getNextInstance().generateStream(prompt, options);
  }

  getName(): string {
    return `${this.instances[0].getName()} (Load Balanced x${this.instances.length})`;
  }
}


export class LLMFactory {
  static getProvider(type?: ProviderType): ILLMProvider {
    const providerType = type || (process.env.LLM_PROVIDER as ProviderType) || 'gemini';

    switch (providerType) {
      case 'gemini': {
        const keysEnv = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
        const keys = keysEnv.split(',').map(k => k.trim()).filter(k => k.length > 0);
        
        if (keys.length === 0) {
          throw new Error("GEMINI_API_KEYS is missing in environment variables");
        }
        
        const instances = keys.map(key => new GeminiProvider(key));
        return instances.length > 1 ? new LoadBalancerProvider(instances) : instances[0];
      }
      default:
        throw new Error(`Unsupported provider type: ${providerType}`);
    }
  }
}

export { GeminiProvider };
