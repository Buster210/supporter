import type {
  Content,
  GenerateContentConfig,
  GenerateContentResponse,
  Tool,
} from "@google/genai";
import type { ToolRegistry } from "./agent";
import { logger } from "./logger";
import { GeminiProvider } from "./providers/gemini-provider";

export interface LLMOptions extends GenerateContentConfig {
  history?: Content[];
  model?: string;
  tools?: Tool[];
  registry?: ToolRegistry;
  interactionId?: string;
  useSearch?: boolean;
  useCodeExecution?: boolean;
}

export function isRateLimit(error: any): boolean {
  const status = error?.status || error?.response?.status;
  const message = error?.message?.toLowerCase() || "";
  return (
    status === 429 ||
    message.includes("429") ||
    message.includes("quota") ||
    message.includes("too many requests")
  );
}

export function isModelError(error: any): boolean {
  const status = error?.status || error?.response?.status;
  const message = error?.message?.toLowerCase() || "";
  return (
    [404, 503, 500].includes(status) ||
    message.includes("unavailable") ||
    message.includes("overloaded") ||
    message.includes("503") ||
    message.includes("404") ||
    message.includes("500") ||
    message.includes("internal error") ||
    message.includes("service level")
  );
}

export function isFallbackError(error: any): boolean {
  return isRateLimit(error) || isModelError(error);
}

export interface LLMResult extends GenerateContentResponse {
  text: string;
  model?: string;
  duration?: number;
  interactionId?: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  raw?: unknown;
}

export interface LLMChunk {
  text: string;
  isLast: boolean;
  raw?: unknown;
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
    let lastError: any;
    const maxRetries = this.instances.length;

    for (let i = 0; i < maxRetries; i++) {
      const instance = this.getNextInstance();
      try {
        return await instance.generate(prompt, options);
      } catch (error) {
        lastError = error;
        if (isRateLimit(error)) {
          continue;
        }
        throw error;
      }
    }
    throw lastError;
  }

  async *generateStream(
    prompt: string | Content[],
    options?: LLMOptions,
  ): AsyncIterable<LLMChunk> {
    yield* this.getNextInstance().generateStream(prompt, options);
  }

  getName(): string {
    const baseName = this.instances[0].getName();
    return this.instances.length > 1
      ? `${baseName} (Round Robin x${this.instances.length})`
      : baseName;
  }
}

export class FallbackProvider implements LLMProvider {
  private primary: LLMProvider;
  private fallback: LLMProvider;

  constructor(primary: LLMProvider, fallback: LLMProvider) {
    this.primary = primary;
    this.fallback = fallback;
  }

  private shouldFallback(error: any): boolean {
    return isFallbackError(error);
  }

  async generate(
    prompt: string | Content[],
    options?: LLMOptions,
  ): Promise<LLMResult> {
    try {
      const result = await this.primary.generate(prompt, options);
      return { ...result, model: result.model || this.primary.getName() };
    } catch (error) {
      if (this.shouldFallback(error)) {
        const result = await this.fallback.generate(prompt, options);
        return { ...result, model: result.model || this.fallback.getName() };
      }
      throw error;
    }
  }

  async *generateStream(
    prompt: string | Content[],
    options?: LLMOptions,
  ): AsyncIterable<LLMChunk> {
    try {
      yield* this.primary.generateStream(prompt, options);
    } catch (error) {
      if (this.shouldFallback(error)) {
        yield* this.fallback.generateStream(prompt, options);
        return;
      }
      throw error;
    }
  }

  getName(): string {
    return `${this.primary.getName()} -> ${this.fallback.getName()}`;
  }
}

/**
 * Resolves the LLM provider based on environment configuration.
 * Supports Round-Robin load balancing over multiple keys and fallback model logic.
 */
export function getProvider(type?: ProviderType): LLMProvider {
  delete process.env.GOOGLE_API_KEY;

  const targetType =
    type || (process.env.LLM_PROVIDER as ProviderType) || "gemini";

  logger.debug`Resolving provider for type: ${targetType}`;

  if (targetType !== "gemini") {
    logger.error`Unsupported provider type: ${targetType}`;
    throw new Error(`Unsupported provider type: ${targetType}`);
  }

  const rawKeys =
    process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || "";
  const keys = rawKeys
    .split(",")
    .map((k) => k.trim())
    .filter(Boolean);

  if (keys.length === 0) {
    logger.error("GEMINI_API_KEYS is missing in environment variables");
    throw new Error("GEMINI_API_KEYS is missing in environment variables");
  }

  const createProvider = (modelName?: string) => {
    const instances = keys.map((key) => {
      const p = new GeminiProvider(key);
      if (modelName) {
        (p as any).modelName = modelName;
      }
      return p;
    });
    return instances.length > 1
      ? new RoundRobinKeyProvider(instances)
      : instances[0];
  };

  const primaryModel =
    process.env.GEMINI_MODEL || "gemini-3.1-flash-lite-preview";
  const fallbackModel = process.env.GEMINI_FALLBACK_MODEL;

  const primary = createProvider(primaryModel);

  if (fallbackModel) {
    logger.info`Using primary model: ${primaryModel} with fallback: ${fallbackModel}`;
    const fallback = createProvider(fallbackModel);
    return new FallbackProvider(primary, fallback);
  }

  logger.info`Using primary model: ${primaryModel}`;
  return primary;
}

export const LLMFactory = {
  getProvider,
};

export { ChatAgent } from "./agent";
export { GeminiProvider };
