import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { LLMFactory } from "../src/index";

describe("LLMFactory Intelligence", () => {
  const originalEnv = { ...process.env };

  beforeAll(() => {
    process.env.GEMINI_API_KEY = "test-key";
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("should default to gemini if no type or ENV is provided", () => {
    delete process.env.LLM_PROVIDER;
    const provider = LLMFactory.getProvider();
    expect(provider.getName()).toContain("gemini");
  });

  test("should detect provider type from LLM_PROVIDER environment variable", () => {
    process.env.LLM_PROVIDER = "gemini";
    const provider = LLMFactory.getProvider();
    expect(provider.getName()).toContain("gemini");
  });

  test("should throw error for unsupported provider type", () => {
    expect(() => LLMFactory.getProvider("unsupported" as any)).toThrow(/Unsupported provider type/);
  });
});
