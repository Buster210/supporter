import { GeminiWrapper } from "./gemini-wrapper";

async function main() {
  try {
    const wrapper = new GeminiWrapper();
    const prompts = ["Hi", "Hello", "How are you?"];

    console.log(`Using model: ${wrapper.getModelName()}`);
    console.log(`Sending ${prompts.length} short requests to test load balancing...`);

    for (let i = 0; i < prompts.length; i++) {
      const prompt = prompts[i];
      console.log(`\n--- Request ${i + 1} ---`);
      console.log(`Prompt: "${prompt}"`);

      const startTime = performance.now();
      let firstTokenTime: number | null = null;

      const streamResult = await wrapper.generateContentStream(prompt);

      process.stdout.write("Response: ");

      let usage: any = null;

      for await (const chunk of streamResult) {
        if (firstTokenTime === null) {
          firstTokenTime = performance.now();
        }
        process.stdout.write(chunk.text || "");

        if (chunk.usageMetadata) {
          usage = chunk.usageMetadata;
        }
      }

      const endTime = performance.now();
      const ttft = firstTokenTime ? ((firstTokenTime - startTime) / 1000).toFixed(2) : "N/A";
      const totalDuration = ((endTime - startTime) / 1000).toFixed(2);

      console.log(`\n\n[Metrics] TTFT: ${ttft}s | Total: ${totalDuration}s`);

      if (usage) {
        console.log(`[Usage] Prompt: ${usage.promptTokenCount} | Response: ${usage.candidatesTokenCount} | Total: ${usage.totalTokenCount}`);
      }
    }

  } catch (error: any) {
    console.error("Test failed:", error.message);
  }
}

main();
