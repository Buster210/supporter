import type {
  Content,
  GenerateContentRequest,
  GenerateContentResponse,
  Schema,
} from "@google/genai";
import { GeminiProvider } from "./providers/gemini-provider";

export interface LLMOptions
  extends Partial<Omit<GenerateContentRequest, "contents">> {
  history?: Content[];
  systemInstruction?: string;
  responseSchema?: Schema;
}

export interface LLMResult extends GenerateContentResponse {
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

export interface LLMProvider {
  generate(
    prompt: string | Content[],
    options?: LLMOptions,
  ): Promise<LLMResult>;
  generateStream(
    prompt: string | Content[],
    options?: LLMOptions,
  ): AsyncIterable<LLMChunk>;
  getName(): string;
}

export type ProviderType = "gemini" | "openai" | "local" | "openrouter";

export class RoundRobinKeyProvider implements LLMProvider {
  private instances: LLMProvider[];
  private currentIndex: number = 0;

  constructor(instances: LLMProvider[]) {
    if (instances.length === 0) {
      throw new Error(
        "RoundRobinKeyProvider requires at least one provider instance.",
      );
    }
    this.instances = instances;
  }

  private getNextInstance(): LLMProvider {
    const instance = this.instances[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.instances.length;
    return instance;
  }

  async generate(
    prompt: string | Content[],
    options?: LLMOptions,
  ): Promise<LLMResult> {
    return this.getNextInstance().generate(prompt, options);
  }

  async *generateStream(
    prompt: string | Content[],
    options?: LLMOptions,
  ): AsyncIterable<LLMChunk> {
    yield* this.getNextInstance().generateStream(prompt, options);
  }

  getName(): string {
    return `${this.instances[0].getName()} (Round Robin x${this.instances.length})`;
  }
}

export class LLMFactory {
  static getProvider(type?: ProviderType): LLMProvider {
    const targetType =
      type || (process.env.LLM_PROVIDER as ProviderType) || "gemini";

    if (targetType !== "gemini") {
      throw new Error(`Unsupported provider type: ${targetType}`);
    }

    const rawKeys =
      process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || "";
    const keys = rawKeys
      .split(",")
      .map((k) => k.trim())
      .filter(Boolean);

    if (keys.length === 0) {
      throw new Error("GEMINI_API_KEYS is missing in environment variables");
    }

    const instances = keys.map((key) => new GeminiProvider(key));
    return instances.length > 1
      ? new RoundRobinKeyProvider(instances)
      : instances[0];
  }
}

export { ChatAgent } from "./agent";
export { GeminiProvider };
