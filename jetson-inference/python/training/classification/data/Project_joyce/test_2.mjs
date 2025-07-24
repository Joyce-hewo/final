import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // Make sure your API key is set in the environment
});

const response = await openai.chat.completions.create({
  model: "gpt-4o", // or "gpt-4" or "gpt-3.5-turbo"
  messages: [
    { role: "user", content: "Write a one-sentence bedtime story about a unicorn." },
  ],
  temperature: 0.7,
});

console.log(response.choices[0].message.content);
