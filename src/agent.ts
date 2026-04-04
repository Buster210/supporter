import type { Content, Tool } from "@google/genai";
import type { LLMProvider } from "./index";
import { logger } from "./logger";

export interface ToolRegistry {
  [name: string]: (args: any) => Promise<any> | any;
}

export interface AgentOptions {
  tools?: Tool[];
  registry?: ToolRegistry;
  systemInstruction?: string;
}

export class ChatAgent {
  private history: Content[] = [];
  private currentInteractionId?: string;
  private provider: LLMProvider;
  private tools?: Tool[];
  private registry?: ToolRegistry;
  private systemInstruction?: string;

  constructor(provider: LLMProvider, options?: AgentOptions) {
    this.provider = provider;
    this.tools = options?.tools;
    this.registry = options?.registry;
    this.systemInstruction = options?.systemInstruction;
  }

/**
 * Executes a prompt against the configured provider.
 * Manages history synchronization between automatic tool calls and simple text responses.
 */
  async execute(prompt: string): Promise<{ text: string; model?: string; duration?: number }> {
    logger.debug`Executing prompt: ${prompt}`;
    const userMessage: Content = { role: "user", parts: [{ text: prompt }] };
    
    const result = await this.provider.generate(prompt, {
      history: this.history,
      interactionId: this.currentInteractionId,
      tools: this.tools,
      registry: this.registry,
      systemInstruction: this.systemInstruction,
    });

    this.currentInteractionId = result.interactionId;

    if (result.automaticFunctionCallingHistory) {
      logger.debug("Updating history with automatic function calling results");
      this.history = result.automaticFunctionCallingHistory;
    } else {
      this.history.push(userMessage);
      this.history.push({
        role: "model",
        parts: result.candidates?.[0]?.content?.parts || [],
      });
    }

    logger.debug`Execution complete. Response length: ${result.text?.length || 0}`;
    return { text: result.text || "", model: result.model, duration: result.duration };
  }

  getHistory(): Content[] {
    return this.history;
  }

  clearHistory(): void {
    logger.info("Clearing agent history");
    this.history = [];
  }
}
