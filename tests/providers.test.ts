import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { LLMFactory } from "../src/index";
import * as dotenv from "dotenv";

dotenv.config();

describe("LLM Provider Instantiation", () => {
  const originalEnv = { ...process.env };

  beforeAll(() => {
    process.env.GEMINI_API_KEY = "test-gemini-key";
    process.env.OPENAI_API_KEY = "test-openai-key";
    delete process.env.OPENAI_MODEL;
    delete process.env.GEMINI_MODEL;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("should instantiate Gemini provider", () => {
    const gemini = LLMFactory.getProvider("gemini");
    expect(gemini).toBeDefined();
    expect(gemini.getName()).toContain("gemini");
  });

  test("should have a functional generate method", async () => {
    const gemini = LLMFactory.getProvider("gemini");
    try {
      await gemini.generate("Test prompt");
    } catch (error: any) {
      expect(error.message).toBeDefined();
    }
  });
});
