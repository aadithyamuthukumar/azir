import { config } from "dotenv";
import OpenAI from "openai";
import { Azir } from "azir";

config();

const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
  throw new Error("OPENAI_API_KEY is required");
}

const openai = new OpenAI({ apiKey: openaiApiKey });
const azir = new Azir({
  openai,
  azirUrl: process.env.AZIR_URL ?? "http://localhost:8080",
});

const response = await azir.responses.create({
  model: "gpt-5-mini",
  feature: "summarization",
  input:
    "Summarize this in one sentence: Azir tracks LLM token usage and cost so teams can find expensive prompts.",
});

console.log(response.output_text);
