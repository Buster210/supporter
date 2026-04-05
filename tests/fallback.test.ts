import { describe, expect, mock, test } from "bun:test";
import { FallbackProvider, type LLMProvider, type LLMResult } from "../src/index";

describe("FallbackProvider", () => {
  test("should switch to fallback if primary fails with 503", async () => {
    const primary: LLMProvider = {
      getName: () => "primary",
      generate: mock().mockRejectedValue({ status: 503, message: "UNAVAILABLE" }),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const fallback: LLMProvider = {
      getName: () => "fallback",
      generate: mock().mockResolvedValue({ text: "Success from fallback" } as LLMResult),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const provider = new FallbackProvider(primary, fallback);
    const result = await provider.generate("test");

    expect(result.text).toBe("Success from fallback");
    expect(primary.generate).toHaveBeenCalled();
    expect(fallback.generate).toHaveBeenCalled();
  });

  test("should not switch to fallback on unrecognized error", async () => {
    const primary: LLMProvider = {
      getName: () => "primary",
      generate: mock().mockRejectedValue(new Error("Generic error")),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const fallback: LLMProvider = {
      getName: () => "fallback",
      generate: mock().mockResolvedValue({ text: "Should not reach here" } as LLMResult),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const provider = new FallbackProvider(primary, fallback);
    
    try {
      await provider.generate("test");
      throw new Error("Should have thrown");
    } catch (e: any) {
      expect(e.message).toBe("Generic error");
    }

    expect(fallback.generate).not.toHaveBeenCalled();
  });
});
