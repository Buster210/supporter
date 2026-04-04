import { expect, mock, test } from "bun:test";
import { mockGenAI } from "./mocks";

mock.module("@google/genai", () => ({
  GoogleGenAI: class {
    models = mockGenAI.models;
  },
}));

import * as dotenv from "dotenv";
import { LLMFactory } from "../src/index";

dotenv.config();

test("Provider Streaming Logic", async () => {
  const provider = LLMFactory.getProvider("gemini");
  const prompt = "Say 'Test Success'";

  const streamResult = provider.generateStream(prompt);
  let fullText = "";

  for await (const chunk of streamResult) {
    fullText += chunk.text;
  }

  expect(fullText).toBe("Chunk 1Chunk 2");
  expect(mockGenAI.models.generateContentStream).toHaveBeenCalled();
});
