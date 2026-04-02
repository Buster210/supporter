import { GoogleGenAI } from "@google/genai";
import * as dotenv from "dotenv";
dotenv.config();

async function inspectResponse() {
  const apiKey = process.env.GEMINI_API_KEY;
  const genAI = new GoogleGenAI(apiKey!);
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const result = await model.generateContent("Say hi");
  console.log("Usage Metadata:", JSON.stringify(result.response.usageMetadata, null, 2));
  
  // Checking for headers or rate limit info
  // The SDK doesn't obviously expose headers in the GenerateContentResponse
  console.log("Response Keys:", Object.keys(result.response));
}

inspectResponse();
