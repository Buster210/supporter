import { test, expect, mock } from "bun:test";
import { mockGenAI } from "./mocks";

mock.module("@google/genai", () => ({
  GoogleGenAI: class {
    models = mockGenAI.models;
  },
}));

import { LLMFactory } from "../src/index";
import * as dotenv from "dotenv";

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
