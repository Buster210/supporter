import { GeminiWrapper } from "../src/gemini-wrapper";
import * as dotenv from "dotenv";

dotenv.config();

async function testLoadBalancer() {
  console.log("--- Testing Gemini Key Load Balancer ---");
  
  const wrapper = new GeminiWrapper();
  const prompts = ["Hi", "Hello", "Hey"];

  for (let i = 0; i < prompts.length; i++) {
    console.log(`\nRequest ${i + 1} with prompt: "${prompts[i]}"`);
    try {
      // We don't necessarily need to WAIT for the full response if we just want to verify index rotation
      // but for completeness, let's try a small request.
      // NOTE: If the keys are dummy, this will fail, but we'll see the index logging first.
      await wrapper.generateContent(prompts[i]);
    } catch (error) {
       // Expecting error if keys are dummy
       console.log("Request finished (error expected if using placeholder keys).");
    }
  }
}

testLoadBalancer().catch(console.error);
