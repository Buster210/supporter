import { GoogleGenAI } from "@google/genai";
import { ILLMProvider, LLMOptions, LLMResult, LLMChunk } from "../index";

export class GeminiProvider implements ILLMProvider {
  private client: GoogleGenAI;
  private modelName: string;
  private defaultSystemInstruction: string;

  constructor(apiKey: string, systemInstruction?: string) {
    this.client = new GoogleGenAI({ apiKey });
    this.modelName = process.env.GEMINI_MODEL || "gemini-3.1-flash-lite-preview";
    this.defaultSystemInstruction = systemInstruction || 
      "You are a helpful assistant. Prioritize quality and clarity in every response.";
  }

  async generate(prompt: string, options?: LLMOptions): Promise<LLMResult> {
    try {
      const result = await this.client.models.generateContent({
        model: options?.model || this.modelName,
        contents: prompt,
        config: {
          systemInstruction: options?.systemInstruction || this.defaultSystemInstruction,
          temperature: options?.temperature ?? 0.7,
          topP: options?.topP,
          topK: options?.topK,
          maxOutputTokens: options?.maxOutputTokens,
        }
      });

      return {
        text: result.text || "",
        usage: result.usageMetadata ? {
          promptTokens: result.usageMetadata.promptTokenCount,
          completionTokens: result.usageMetadata.candidatesTokenCount,
          totalTokens: result.usageMetadata.totalTokenCount,
        } : undefined,
        raw: result
      };
    } catch (error) {
      console.error("Error calling Gemini API:", error);
      throw error;
    }
  }

  async *generateStream(prompt: string, options?: LLMOptions): AsyncIterable<LLMChunk> {
    try {
      const stream = await this.client.models.generateContentStream({
        model: options?.model || this.modelName,
        contents: prompt,
        config: {
          systemInstruction: options?.systemInstruction || this.defaultSystemInstruction,
          temperature: options?.temperature ?? 0.7,
          topP: options?.topP,
          topK: options?.topK,
        }
      });

      for await (const chunk of stream) {
        yield {
          text: chunk.text || "",
          isLast: false,
          raw: chunk
        };
      }
    } catch (error) {
      console.error("Error calling Gemini API (Stream):", error);
      throw error;
    }
  }

  getName(): string {
    return this.modelName;
  }
}
