import { type Content, GoogleGenAI, type Tool } from "@google/genai";
import type { LLMChunk, LLMOptions, LLMProvider, LLMResult } from "../index";
import { logger } from "../logger";

export class GeminiProvider implements LLMProvider {
  private client: GoogleGenAI;
  private modelName: string;
  private defaultSystemInstruction: string;

  constructor(apiKey: string, systemInstruction?: string) {
    this.client = new GoogleGenAI({
      apiKey,
      httpOptions: {
        retryOptions: {
          attempts: 2, // Low attempts to favor fast key rotation
        },
      },
    });
    this.modelName =
      process.env.GEMINI_MODEL || "gemini-3.1-flash-lite-preview";
    this.defaultSystemInstruction =
      systemInstruction ||
      "You are a helpful assistant. Prioritize quality and clarity in every response.";
  }

  private prepareContents(
    prompt: string | Content[],
    history: Content[] = [],
  ): Content[] {
    const freshContent =
      typeof prompt === "string"
        ? [{ role: "user", parts: [{ text: prompt }] }]
        : prompt;
    return [...history, ...freshContent];
  }

  private transformTools(options: LLMOptions = {}): Tool[] | undefined {
    const { tools, registry, useSearch, useCodeExecution } = options;

    const finalTools: Tool[] = tools ? [...tools] : [];

    if (useSearch) {
      finalTools.push({ googleSearch: {} } as any);
    }

    if (useCodeExecution) {
      finalTools.push({ codeExecution: {} } as any);
    }

    if (!registry) {
      return finalTools.length > 0 ? finalTools : undefined;
    }

    return finalTools.map((tool) => {
      // If it's already a Tool with functionDeclarations, we wrap it in a CallableTool
      if ("functionDeclarations" in tool) {
        return {
          tool: async () => tool,
          callTool: async (calls) => {
            const parts = await Promise.all(
              calls.map(async (call) => {
                const func = registry[call.name];
                if (!func) {
                  logger.error`Tool ${call.name} not found in registry`;
                  return {
                    functionResponse: {
                      name: call.name,
                      response: { error: `Tool ${call.name} not found in registry` },
                    },
                  };
                }
                try {
                  logger.debug`Calling tool: ${call.name} with args: ${JSON.stringify(call.args)}`;
                  const result = await func(call.args);
                  logger.debug`Tool ${call.name} response: ${JSON.stringify(result)}`;
                  return {
                    functionResponse: {
                      name: call.name,
                      response: { result },
                    },
                  };
                } catch (error) {
                  const errorMsg = error instanceof Error ? error.message : String(error);
                  logger.error`Tool ${call.name} failed: ${errorMsg}`;
                  return {
                    functionResponse: {
                      name: call.name,
                      response: { error: errorMsg },
                    },
                  };
                }
              })
            );
            return parts;
          },
        } as unknown as Tool; // The SDK types allow this through CallableTool
      }
      return tool;
    });
  }

  async generate(
    prompt: string | Content[],
    options?: LLMOptions,
  ): Promise<LLMResult> {
    try {
      const {
        history,
        systemInstruction,
        temperature,
        topP,
        topK,
        maxOutputTokens,
        config: userConfig,
        tools,
        registry,
        ...sdkOptions
      } = options || {};

      const transformedTools = this.transformTools(options);
      logger.debug`Generating content for model: ${sdkOptions.model || this.modelName}`;
      const contents = this.prepareContents(prompt, history);
      const systemInstructionContent = systemInstruction || this.defaultSystemInstruction;

      let result: any;

      const startTime = performance.now();
      if (options?.interactionId) {
        try {
          result = await this.client.interactions.create({
            model: sdkOptions.model || this.modelName,
            input: typeof prompt === "string" ? prompt : JSON.stringify(prompt),
            previous_interaction_id: options.interactionId,
            tools: transformedTools,
            config: {
              systemInstruction: systemInstructionContent,
              automaticFunctionCalling: transformedTools ? { disable: false } : undefined,
              ...userConfig,
            },
          });
        } catch (err) {
          logger.warn(`[Interaction] Failed to continue interaction ${options.interactionId}. Falling back to content generation.`, err);
          // Fallback to generateContent handled below by letting result be null
        }
      }

      if (!result) {
        result = await this.client.models.generateContent({
          model: sdkOptions.model || this.modelName,
          contents,
          tools: transformedTools,
          config: {
            systemInstruction: systemInstructionContent,
            temperature: temperature ?? userConfig?.temperature ?? 0.7,
            topP: topP ?? userConfig?.topP,
            topK: topK ?? userConfig?.topK,
            maxOutputTokens: maxOutputTokens ?? userConfig?.maxOutputTokens,
            automaticFunctionCalling: transformedTools ? { disable: false } : undefined,
            ...userConfig,
          },
          ...sdkOptions,
        });
      }
      const endTime = performance.now();

      return {
        ...result,
        text: result.text || "",
        model: sdkOptions.model || this.modelName,
        duration: (endTime - startTime) / 1000,
        interactionId: result.id, // SDK Interactions have .id
        usage: result.usageMetadata && {
          promptTokens: result.usageMetadata.promptTokenCount || 0,
          completionTokens: result.usageMetadata.candidatesTokenCount || 0,
          totalTokens: result.usageMetadata.totalTokenCount || 0,
        },
        raw: result,
      } as LLMResult;
    } catch (error) {
      logger.error(`Error in generate: ${error instanceof Error ? error.stack : String(error)}`);
      throw error;
    }
  }

  async *generateStream(
    prompt: string | Content[],
    options?: LLMOptions,
  ): AsyncIterable<LLMChunk> {
    try {
      const {
        history,
        systemInstruction,
        temperature,
        topP,
        topK,
        maxOutputTokens,
        config: userConfig,
        tools,
        registry,
        ...sdkOptions
      } = options || {};

      const transformedTools = this.transformTools(options);

      const stream = await this.client.models.generateContentStream({
        model: sdkOptions.model || this.modelName,
        contents: this.prepareContents(prompt, history),
        tools: transformedTools,
        config: {
          systemInstruction: systemInstruction || this.defaultSystemInstruction,
          temperature: temperature ?? userConfig?.temperature ?? 0.7,
          topP: topP ?? userConfig?.topP,
          topK: topK ?? userConfig?.topK,
          maxOutputTokens: maxOutputTokens ?? userConfig?.maxOutputTokens,
          automaticFunctionCalling: transformedTools ? { disable: false } : undefined,
          ...userConfig,
        },
        ...sdkOptions,
      });

      for await (const chunk of stream) {
        yield {
          text: chunk.text || "",
          isLast: false,
          raw: chunk,
        };
      }
    } catch (error) {
      logger.error(`Error in generateStream: ${error instanceof Error ? error.stack : String(error)}`);
      throw error;
    }
  }

  getName(): string {
    return this.modelName;
  }
}
