import { GeminiWrapper } from "./gemini-wrapper";

async function main() {
  try {
    const wrapper = new GeminiWrapper();
    const prompt = "Explain how AI works in detail";

    console.log(`Using model: ${wrapper.getModelName()}`);
    console.log(`Sending prompt: "${prompt}"...`);

    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    
    const streamResult = await wrapper.generateContentStream(prompt);

    process.stdout.write("\n--- Response Stream ---\n");
    
    let fullText = "";
    for await (const chunk of streamResult) {
      if (firstTokenTime === null) {
        firstTokenTime = performance.now();
      }
      const chunkText = chunk.text || "";
      fullText += chunkText;
      process.stdout.write(chunkText);
    }

    const endTime = performance.now();
    const ttft = firstTokenTime ? ((firstTokenTime - startTime) / 1000).toFixed(2) : "N/A";
    const totalDuration = ((endTime - startTime) / 1000).toFixed(2);

    console.log(`\n\n--- Metrics ---`);
    console.log(`⏱️ Time to First Token (TTFT): ${ttft}s`);
    console.log(`✨ Total response time: ${totalDuration}s`);
    console.log(`📏 Character count: ${fullText.length}`);

  } catch (error: any) {
    console.error("Test failed:", error.message);
  }
}

main();
