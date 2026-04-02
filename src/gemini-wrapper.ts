import { GoogleGenAI } from "@google/genai";
import * as dotenv from "dotenv";
dotenv.config();

export class GeminiWrapper {
  private clients: GoogleGenAI[] = [];
  private modelName: string;
  private systemInstruction: string;
  private currentKeyIndex: number = 0;

  constructor(systemInstruction?: string) {
    const apiKeys = process.env.GEMINI_API_KEYS?.split(",").map(key => key.trim()).filter(key => key.length > 0);
    
    if (!apiKeys || apiKeys.length === 0) {
      throw new Error("GEMINI_API_KEYS is missing or empty in .env file");
    }

    this.clients = apiKeys.map(apiKey => new GoogleGenAI({ apiKey }));
    this.modelName = process.env.GEMINI_MODEL || "gemini-3.1-flash-lite-preview";
    this.systemInstruction = systemInstruction || 
      "You are a helpful assistant. Prioritize quality and clarity in every response.";
    
    console.log(`Initialized GeminiWrapper with ${this.clients.length} API keys.`);
  }

  private getNextClient() {
    const client = this.clients[this.currentKeyIndex];
    console.log(`[Load Balancer] Using API Key at index: ${this.currentKeyIndex}`);
    this.currentKeyIndex = (this.currentKeyIndex + 1) % this.clients.length;
    return client;
  }

  async generateContent(prompt: string) {
    try {
      const client = this.getNextClient();
      const result = await client.models.generateContent({
        model: this.modelName,
        contents: prompt,
        config: {
          systemInstruction: this.systemInstruction,
          temperature: 0.7,
        }
      });
      return result;
    } catch (error) {
      console.error(`Error calling Gemini API with key index ${this.currentKeyIndex}:`, error);
      throw error;
    }
  }

  async generateContentStream(prompt: string) {
    try {
      const client = this.getNextClient();
      return await client.models.generateContentStream({
        model: this.modelName,
        contents: prompt,
        config: {
          systemInstruction: this.systemInstruction,
          temperature: 0.7,
          topP: 0.8,
          topK: 40,
        }
      });
    } catch (error) {
      console.error(`Error calling Gemini API (Stream) with key index ${this.currentKeyIndex}:`, error);
      throw error;
    }
  }

  getModelName(): string {
    return this.modelName;
  }
}
