import { describe, expect, mock, test } from "bun:test";
import { ChatAgent, type LLMProvider } from "../src/index";

describe("ChatAgent Execution Loop", () => {
  test("should execute tool and provide final answer", async () => {
    const mockProvider: LLMProvider = {
      getName: () => "mock-provider",
      generate: mock()
        .mockResolvedValueOnce({
          text: "",
          candidates: [
            {
              content: {
                parts: [
                  {
                    functionCall: {
                      name: "get_weather",
                      args: { location: "London" },
                    },
                  },
                ],
              },
            },
          ],
        } as LLMResult)
        .mockResolvedValueOnce({
          text: "The weather in London is sunny.",
          candidates: [
            {
              content: {
                parts: [{ text: "The weather in London is sunny." }],
              },
            },
          ],
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

    expect(response).toBe("The weather in London is sunny.");
    expect(mockProvider.generate).toHaveBeenCalledTimes(2);

    const history = agent.getHistory();
    // 1: User prompt
    // 2: Model function call
    // 3: User function response
    // 4: Model final answer
    expect(history.length).toBe(4);
    const secondPart = history[2].parts[0];
    if ("functionResponse" in secondPart) {
      expect(secondPart.functionResponse.name).toBe("get_weather");
    } else {
      throw new Error("Expected functionResponse in part");
    }
  });
});
