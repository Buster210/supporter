import { describe, expect, mock, test } from "bun:test";
import {
  type LLMProvider,
  type LLMResult,
  RoundRobinKeyProvider,
} from "../src/index";

describe("Smart RoundRobin Retries", () => {
  test("should retry on 429 (Rate Limit)", async () => {
    let callCount = 0;
    const mock1: LLMProvider = {
      getName: () => "key1",
      generate: mock().mockImplementation(async () => {
        callCount++;
        const err: any = new Error("Quota exceeded");
        err.status = 429;
        throw err;
      }),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const mock2: LLMProvider = {
      getName: () => "key2",
      generate: mock().mockImplementation(async () => {
        callCount++;
        return { text: "Success from Key 2" } as LLMResult;
      }),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const rr = new RoundRobinKeyProvider([mock1, mock2]);
    const result = await rr.generate("test prompt");

    expect(result.text).toBe("Success from Key 2");
    expect(callCount).toBe(2);
  });

  test("should fast-fail on 503 (Model Unavailable)", async () => {
    let callCount = 0;
    const mock1: LLMProvider = {
      getName: () => "key1",
      generate: mock().mockImplementation(async () => {
        callCount++;
        const err: any = new Error("Service Unavailable");
        err.status = 503;
        throw err;
      }),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const mock2: LLMProvider = {
      getName: () => "key2",
      generate: mock().mockImplementation(
        mock().mockResolvedValue({ text: "Should not be called" }),
      ),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const rr = new RoundRobinKeyProvider([mock1, mock2]);

    try {
      await rr.generate("test prompt");
      throw new Error("Should have thrown");
    } catch (e: any) {
      expect(e.status).toBe(503);
    }

    expect(callCount).toBe(1); // Should NOT have retried Key 2
  });
});
