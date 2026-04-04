import { mock } from "bun:test";

export const getMockGeminiResponse = (text: string = "Mocked Response") => {
  const usageMetadata = {
    promptTokenCount: 10,
    candidatesTokenCount: 20,
    totalTokenCount: 30,
  };
  return {
    text,
    usageMetadata,
    usage: usageMetadata
  };
};

export const mockGenAI = {
  models: {
    generateContent: mock(async () => getMockGeminiResponse()),
    generateContentStream: mock(async function* () {
      yield { text: "Chunk 1", usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 5, totalTokenCount: 10 } };
      yield { text: "Chunk 2", usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 10, totalTokenCount: 15 } };
    }),
  },
};
