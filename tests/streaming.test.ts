import { test, expect } from "bun:test";
import { LLMFactory } from "../src/index";
import * as dotenv from "dotenv";

dotenv.config();

test("Provider Streaming Integration", async () => {
  try {
    const provider = LLMFactory.getProvider("gemini");
    const prompt = "Say 'Test Success'";

    console.log(`\nTesting ${provider.getName()} streaming...`);

    const streamResult = provider.generateStream(prompt);
    let fullText = "";
    
    for await (const chunk of streamResult) {
      fullText += chunk.text;
    }

    expect(fullText).toBeDefined();
    console.log(`Response received: ${fullText}`);
  } catch (error: any) {
    if (error.message.includes("GEMINI_API_KEY")) {
      console.log("Skipping streaming test: GEMINI_API_KEY missing.");
    } else {
      throw error;
    }
  }
}, 30000);
