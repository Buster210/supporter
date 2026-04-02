import { GoogleGenAI } from "@google/genai";
import * as dotenv from "dotenv";
dotenv.config();

export class GeminiWrapper {
  private client: GoogleGenAI;
  private modelName: string;
  private systemInstruction: string;

  constructor(systemInstruction?: string) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is missing in .env file");
    }

    this.client = new GoogleGenAI({ apiKey });
    this.modelName = process.env.GEMINI_MODEL || "gemini-3.1-flash-lite-preview";
    this.systemInstruction = systemInstruction || 
      "You are a helpful assistant. Prioritize quality and clarity in every response.";
  }

  async generateContent(prompt: string) {
    try {
      const result = await this.client.models.generateContent({
        model: this.modelName,
        contents: prompt,
        config: {
          systemInstruction: this.systemInstruction,
          temperature: 0.7,
        }
      });
      return result;
    } catch (error) {
      console.error("Error calling Gemini API:", error);
      throw error;
    }
  }

  async generateContentStream(prompt: string) {
    try {
      return await this.client.models.generateContentStream({
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
      console.error("Error calling Gemini API (Stream):", error);
      throw error;
    }
  }

  getModelName(): string {
    return this.modelName;
  }
}
