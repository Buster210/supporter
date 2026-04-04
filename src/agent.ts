import { Content, Tool, Schema } from "@google/genai";
import { ILLMProvider, LLMResult } from "./index";

export interface ToolRegistry {
  [name: string]: (...args: any[]) => Promise<any> | any;
}

export interface AgentOptions {
  tools?: Tool[];
  registry?: ToolRegistry;
  systemInstruction?: string;
}

export class ChatAgent {
  private history: Content[] = [];
  private provider: ILLMProvider;
  private tools?: Tool[];
  private registry?: ToolRegistry;
  private systemInstruction?: string;

  constructor(provider: ILLMProvider, options?: AgentOptions) {
    this.provider = provider;
    this.tools = options?.tools;
    this.registry = options?.registry;
    this.systemInstruction = options?.systemInstruction;
  }

  async execute(prompt: string): Promise<string> {
    this.history.push({ role: "user", parts: [{ text: prompt }] });

    while (true) {
      const result = await this.provider.generate(this.history, {
        tools: this.tools,
        systemInstruction: this.systemInstruction,
      });

      this.history.push({
        role: "model",
        parts: result.candidates?.[0]?.content?.parts || []
      });

      const calls = result.candidates?.[0]?.content?.parts?.filter(p => p.functionCall) || [];
      if (calls.length === 0 || !this.registry) {
        return result.text || "";
      }

      const toolParts = await Promise.all(
        calls.map(async (part) => {
          const call = part.functionCall!;
          const tool = this.registry![call.name];
          
          if (!tool) return null;

          try {
            const response = await tool(call.args);
            return {
              functionResponse: {
                name: call.name,
                response: { result: response }
              }
            };
          } catch (error: any) {
            return {
              functionResponse: {
                name: call.name,
                response: { error: error.message || "Unknown error" }
              }
            };
          }
        })
      );

      const validParts = toolParts.filter(Boolean);
      if (validParts.length === 0) {
        return result.text || "";
      }

      this.history.push({ role: "user", parts: validParts as any[] });
    }
  }

  getHistory(): Content[] {
    return this.history;
  }

  clearHistory(): void {
    this.history = [];
  }
}
