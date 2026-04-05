import { afterAll, beforeAll, describe, expect, mock, test } from "bun:test";
import { getMockGeminiResponse, mockGenAI } from "./mocks";

mock.module("@google/genai", () => ({
  GoogleGenAI: class {
    models = mockGenAI.models;
  },
}));

import * as dotenv from "dotenv";
import { getProvider, LLMFactory } from "../src/index";

dotenv.config();
process.env.GEMINI_API_KEY = "test-streaming-key";

describe("LLM Provider Instantiation & Logic", () => {
  const originalEnv = { ...process.env };

  beforeAll(() => {
    process.env.GEMINI_API_KEY = "test-gemini-key";
    delete process.env.GEMINI_MODEL;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("should instantiate Gemini provider", () => {
    const gemini = getProvider("gemini");
    expect(gemini).toBeDefined();
    expect(gemini.getName()).toContain("gemini");
  });

  test("should transform generate output correctly", async () => {
    const gemini = getProvider("gemini");
    const mockResponse = getMockGeminiResponse("Mock success");

    mockGenAI.models.generateContent.mockResolvedValueOnce(mockResponse);

    const result = await gemini.generate("Test prompt");

    expect(result.text).toBe("Mock success");
    expect(result.usage?.totalTokens).toBe(30);
    expect(mockGenAI.models.generateContent).toHaveBeenCalled();
  });

  test("should pass options to Gemini client", async () => {
    const gemini = LLMFactory.getProvider("gemini");

    await gemini.generate("Test prompt", {
      temperature: 0.1,
      topP: 0.9,
      maxOutputTokens: 100,
    });

    const lastCall = mockGenAI.models.generateContent.mock.calls.at(-1)[0];
    expect(lastCall.config.temperature).toBe(0.1);
    expect(lastCall.config.topP).toBe(0.9);
    expect(lastCall.config.maxOutputTokens).toBe(100);
  });
});
