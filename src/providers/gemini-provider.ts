import { GoogleGenAI, Content } from "@google/genai";
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

  private prepareContents(prompt: string | Content[], history: Content[] = []): Content[] {
    const freshContent = typeof prompt === "string" ? [{ role: "user", parts: [{ text: prompt }] }] : prompt;
    return [...history, ...freshContent];
  }

  async generate(prompt: string | Content[], options?: LLMOptions): Promise<LLMResult> {
    try {
      const { 
        history, 
        systemInstruction, 
        temperature,
        topP,
        topK,
        maxOutputTokens,
        config: userConfig,
        ...sdkOptions 
      } = options || {};
      
      const response = await this.client.models.generateContent({
        model: sdkOptions.model || this.modelName,
        contents: this.prepareContents(prompt, history),
        config: {
          systemInstruction: systemInstruction || this.defaultSystemInstruction,
          temperature: temperature ?? userConfig?.temperature ?? 0.7,
          topP: topP ?? userConfig?.topP,
          topK: topK ?? userConfig?.topK,
          maxOutputTokens: maxOutputTokens ?? userConfig?.maxOutputTokens,
          ...userConfig,
        },
        ...sdkOptions
      });

      return {
        ...response,
        text: response.text || "",
        usage: response.usageMetadata && {
          promptTokens: response.usageMetadata.promptTokenCount || 0,
          completionTokens: response.usageMetadata.candidatesTokenCount || 0,
          totalTokens: response.usageMetadata.totalTokenCount || 0,
        },
        raw: response
      } as LLMResult;
    } catch (error) {
      console.error("Error calling Gemini API:", error);
      throw error;
    }
  }

  async *generateStream(prompt: string | Content[], options?: LLMOptions): AsyncIterable<LLMChunk> {
    try {
      const { 
        history, 
        systemInstruction, 
        temperature,
        topP,
        topK,
        maxOutputTokens,
        config: userConfig,
        ...sdkOptions 
      } = options || {};

      const stream = await this.client.models.generateContentStream({
        model: sdkOptions.model || this.modelName,
        contents: this.prepareContents(prompt, history),
        config: {
          systemInstruction: systemInstruction || this.defaultSystemInstruction,
          temperature: temperature ?? userConfig?.temperature ?? 0.7,
          topP: topP ?? userConfig?.topP,
          topK: topK ?? userConfig?.topK,
          maxOutputTokens: maxOutputTokens ?? userConfig?.maxOutputTokens,
          ...userConfig,
        },
        ...sdkOptions
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
