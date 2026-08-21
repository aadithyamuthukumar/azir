import { createHash } from "node:crypto";
import { config } from "dotenv";
import OpenAI from "openai";

config();

const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
  throw new Error("OPENAI_API_KEY is required");
}

const azirUrl = process.env.AZIR_URL ?? "http://localhost:8080";
const model = "gpt-5-mini";
const feature = "summarization";
const prompt =
  "Summarize this in one sentence: Azir tracks LLM token usage and cost so teams can find expensive prompts.";

const openai = new OpenAI({ apiKey: openaiApiKey });

const startedAt = Date.now();
const response = await openai.responses.create({
  model,
  input: prompt,
});
const latencyMs = Date.now() - startedAt;

const inputTokens = response.usage?.input_tokens;
const outputTokens = response.usage?.output_tokens;
if (inputTokens == null || outputTokens == null) {
  throw new Error("OpenAI response did not include usage token counts");
}

const promptHash = createHash("sha256").update(prompt).digest("hex");

console.log("OpenAI response text:");
console.log(response.output_text);
console.log("\nOpenAI usage:");
console.log({
  input_tokens: inputTokens,
  output_tokens: outputTokens,
  latency_ms: latencyMs,
  prompt_hash: promptHash,
});

const event = {
  model,
  feature,
  input_tokens: inputTokens,
  output_tokens: outputTokens,
  latency_ms: latencyMs,
  prompt_hash: promptHash,
};

const azirResponse = await fetch(`${azirUrl}/v1/events`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(event),
});

const azirBody = await azirResponse.text();
console.log("\nAzir response status:", azirResponse.status);
console.log("Azir response body:");
console.log(azirBody);

if (!azirResponse.ok) {
  process.exitCode = 1;
}
