import { describe, expect, mock, test } from "bun:test";
import type { Content, Tool } from "@google/genai";
import { ChatAgent, type LLMProvider, type LLMResult } from "../src/index";

describe("ChatAgent Execution Loop", () => {
  test("should execute tool and provide final answer", async () => {
    const mockHistory: Content[] = [
      { role: "user", parts: [{ text: "What is the weather in London?" }] },
      {
        role: "model",
        parts: [
          {
            functionCall: { name: "get_weather", args: { location: "London" } },
          },
        ],
      },
      {
        role: "user",
        parts: [
          {
            functionResponse: {
              name: "get_weather",
              response: { result: "Sunny in London" },
            },
          },
        ],
      },
      { role: "model", parts: [{ text: "The weather in London is sunny." }] },
    ];

    const mockProvider: LLMProvider = {
      getName: () => "mock-provider",
      generate: mock().mockResolvedValue({
        text: "The weather in London is sunny.",
        candidates: [
          {
            content: {
              parts: [{ text: "The weather in London is sunny." }],
            },
          },
        ],
        automaticFunctionCallingHistory: mockHistory,
      } as LLMResult),
      generateStream: mock().mockImplementation(async function* () {}),
    };

    const registry = {
      get_weather: async (args: { location: string }) => {
        return `Sunny in ${args.location}`;
      },
    };

    const agent = new ChatAgent(mockProvider, {
      tools: [
        {
          functionDeclarations: [
            {
              name: "get_weather",
              description: "get weather",
              parameters: {
                type: "OBJECT",
                properties: { location: { type: "STRING" } },
              },
            },
          ],
        } as Tool,
      ],
      registry,
    });

    const response = await agent.execute("What is the weather in London?");

    expect(response.text).toBe("The weather in London is sunny.");
    expect(mockProvider.generate).toHaveBeenCalledTimes(1);

    const history = agent.getHistory();
    expect(history.length).toBe(4);
    expect(history[3].parts[0].text).toBe("The weather in London is sunny.");

    const thirdPart = history[2].parts[0];
    if ("functionResponse" in thirdPart) {
      expect(thirdPart.functionResponse.name).toBe("get_weather");
    } else {
      throw new Error("Expected functionResponse in part");
    }
  });
});
