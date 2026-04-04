import { describe, expect, mock, test } from "bun:test";
import {
  type LLMChunk,
  type LLMProvider,
  type LLMResult,
  RoundRobinKeyProvider,
} from "../src/index";

const createMockProvider = (name: string): LLMProvider => ({
  generate: mock(async () => ({ text: `Response from ${name}` }) as LLMResult),
  generateStream: mock(async function* () {
    yield { text: `Stream from ${name}`, isLast: true } as LLMChunk;
  }),
  getName: () => name,
});

describe("RoundRobinKeyProvider", () => {
  test("should cycle through providers in round-robin fashion", async () => {
    const p1 = createMockProvider("P1");
    const p2 = createMockProvider("P2");
    const lb = new RoundRobinKeyProvider([p1, p2]);

    const res1 = await lb.generate("test");
    expect(res1.text).toBe("Response from P1");
    expect(p1.generate).toHaveBeenCalledTimes(1);

    const res2 = await lb.generate("test");
    expect(res2.text).toBe("Response from P2");
    expect(p2.generate).toHaveBeenCalledTimes(1);

    const res3 = await lb.generate("test");
    expect(res3.text).toBe("Response from P1");
    expect(p1.generate).toHaveBeenCalledTimes(2);
  });

  test("should cycle through stream providers", async () => {
    const p1 = createMockProvider("P1");
    const p2 = createMockProvider("P2");
    const lb = new RoundRobinKeyProvider([p1, p2]);

    const stream1 = lb.generateStream("test");
    const chunk1 = (await stream1.next()).value;
    expect(chunk1.text).toBe("Stream from P1");

    const stream2 = lb.generateStream("test");
    const chunk2 = (await stream2.next()).value;
    expect(chunk2.text).toBe("Stream from P2");
  });

  test("should return combined name", () => {
    const p1 = createMockProvider("P1");
    const p2 = createMockProvider("P2");
    const lb = new RoundRobinKeyProvider([p1, p2]);
    expect(lb.getName()).toBe("P1 (Round Robin x2)");
  });
});
