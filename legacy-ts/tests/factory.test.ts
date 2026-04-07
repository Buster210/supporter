import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { getProvider } from "../src/index";

describe("LLMFactory Intelligence", () => {
  const originalEnv = { ...process.env };

  beforeAll(() => {
    delete process.env.GEMINI_FALLBACK_MODEL;
    delete process.env.LLM_PROVIDER;
    delete process.env.GEMINI_API_KEYS;
    process.env.GEMINI_API_KEY = "test-key";
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("should default to gemini if no type or ENV is provided", () => {
    delete process.env.LLM_PROVIDER;
    const provider = getProvider();
    expect(provider.getName()).toContain("gemini");
  });

  test("should detect provider type from LLM_PROVIDER environment variable", () => {
    process.env.LLM_PROVIDER = "gemini";
    const provider = getProvider();
    expect(provider.getName()).toContain("gemini");
  });

  test("should throw error for unsupported provider type", () => {
    expect(() => getProvider("unsupported" as unknown as ProviderType)).toThrow(
      /Unsupported provider type/,
    );
  });

  test("should support multiple API keys via GEMINI_API_KEYS", () => {
    delete process.env.GEMINI_API_KEY;
    process.env.GEMINI_API_KEYS = "key1, key2, key3";
    const provider = getProvider();
    expect(provider.getName()).toContain("Round Robin x3");
  });
});
